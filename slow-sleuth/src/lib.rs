//! Library for clustering spans that do a similar amounts of work together and
//! outputting statistics on timings of each cluster.

use std::{
    collections::HashMap,
    time::{Duration, Instant},
};

use tracing::{Id, Subscriber};
use tracing_subscriber::{registry::LookupSpan, Layer};

/// An in-progress span being recorded.
struct InProgressSpanInfo {
    /// The number of time each child work (event or span) is encountered.
    child_counts: HashMap<&'static str, usize>,
    /// The start time of this span.
    start_time: Instant,
}

/// A completed span of work.
pub struct SpanInfo {
    /// The name of the span (for instrumented functions it's typically the function name).
    pub span_name: &'static str,
    /// The number of time each child work (event or span) is encountered.
    pub child_counts: HashMap<&'static str, usize>,
    /// The duration of time taken by the span.
    pub duration: Duration,
}

/// A consumer that can take completed spans and process them.
pub trait SpanInfoConsumer {
    /// Consume a completed span.
    fn consume_span(&self, span_info: SpanInfo);
}

/// A tracing layer that can track spans and consume them for processing.
pub struct Sleuth<C: SpanInfoConsumer> {
    span_consumer: C,
}

impl<C: SpanInfoConsumer> Sleuth<C> {
    /// Create a new layer with the given span consumer.
    pub fn new(span_consumer: C) -> Sleuth<C> {
        Sleuth { span_consumer }
    }
}

impl<C, S> Layer<S> for Sleuth<C>
where
    S: Subscriber + for<'a> LookupSpan<'a>,
    C: SpanInfoConsumer + 'static,
{
    fn on_event(&self, event: &tracing::Event<'_>, ctx: tracing_subscriber::layer::Context<'_, S>) {
        if let Some(span) = ctx.lookup_current() {
            if let Some(info) = span.extensions_mut().get_mut::<InProgressSpanInfo>() {
                // We're in a span with a tracker - record our work.
                *info
                    .child_counts
                    .entry(event.metadata().name())
                    .or_default() += 1;
            }
        }
    }

    fn on_enter(&self, id: &Id, ctx: tracing_subscriber::layer::Context<'_, S>) {
        if let Some(span) = ctx.span(id) {
            // We're entering a span - start recording what's going on inside it.
            span.extensions_mut().insert(InProgressSpanInfo {
                child_counts: HashMap::default(),
                start_time: Instant::now(),
            });
        }
    }

    fn on_exit(&self, id: &Id, ctx: tracing_subscriber::layer::Context<'_, S>) {
        if let Some(span) = ctx.span(id) {
            if let Some(info) = span.extensions_mut().remove::<InProgressSpanInfo>() {
                // We're exiting a tracked span - check if the span has a parent.
                if let Some(parent) = span.parent() {
                    // There's a parent - just amend the parent's info to record what happened in this span.
                    if let Some(parent_info) =
                        parent.extensions_mut().get_mut::<InProgressSpanInfo>()
                    {
                        // Increment the counts for all the children.
                        for (child, count) in info.child_counts.iter() {
                            *parent_info.child_counts.entry(child).or_default() += count;
                        }
                        // Increment the count for this child span itself.
                        *parent_info.child_counts.entry(span.name()).or_default() += 1;
                    }
                } else {
                    // No parent - that's a root span. Just consume the info.
                    self.span_consumer.consume_span(SpanInfo {
                        span_name: span.name(),
                        child_counts: info.child_counts,
                        duration: info.start_time.elapsed(),
                    });
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;
    use tracing::{event, instrument, Level};
    use tracing_subscriber::{prelude::*, Registry};

    impl SpanInfoConsumer for Arc<Mutex<Vec<SpanInfo>>> {
        fn consume_span(&self, span_info: SpanInfo) {
            self.lock().unwrap().push(span_info);
        }
    }

    #[instrument]
    fn sub_work() {
        event!(Level::INFO, "sub_step");
    }

    #[instrument]
    fn work() {
        event!(Level::INFO, "step1");
        event!(Level::INFO, "step2");
        sub_work();
    }

    #[test]
    fn collection() {
        let collector: Arc<Mutex<Vec<SpanInfo>>> = Arc::new(Mutex::new(vec![]));
        let subscriber = Registry::default().with(Sleuth::new(collector.clone()));
        tracing::subscriber::with_default(subscriber, || {
            work();
            work();
        });
        let obtained = collector.lock().unwrap();
        assert_eq!(2, obtained.len());
        for span in obtained.iter() {
            assert_eq!("work", span.span_name);
            assert_eq!(4, span.child_counts.len()); // 3 events and one span
            assert_eq!(Some(&1), span.child_counts.get("sub_work"));
        }
    }
}
