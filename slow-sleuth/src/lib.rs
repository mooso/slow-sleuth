//! Library for clustering spans that do a similar amounts of work together and
//! outputting statistics on timings of each cluster.

use std::{
    collections::HashMap,
    io::{self, Write},
    sync::mpsc::{self, TrySendError},
    thread::{spawn, JoinHandle},
    time::{Duration, Instant},
};

use clustering::{Cluster, ClusteringParameters, SleuthClassifier};
use itertools::Itertools;
use tracing::{Id, Subscriber};
use tracing_subscriber::{registry::LookupSpan, Layer};

pub mod clustering;

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

/// Parameters for creating a sleuth with a separate thread to handle classification and clustering.
pub struct ClusteringSleuthParameters<W> {
    /// The output for formatted cluster information.
    pub output: W,
    /// The period for outputing cluster information (e.g. if 10 seconds will output every 10 seconds).
    pub output_period: Duration,
    /// The parameters for the clustering algorithm.
    pub clustering_parameters: ClusteringParameters,
    /// Bound on the channel to the thread. If the thread falls behind and this channel fills up, timing info will be discarded.
    pub channel_limit: usize,
}

impl<W> ClusteringSleuthParameters<W> {
    pub fn new(output: W) -> Self {
        Self {
            output,
            output_period: Duration::from_secs(10),
            clustering_parameters: Default::default(),
            channel_limit: 1024,
        }
    }
}

/// Information about the sleuth created with a thread spawned for classifying information.
pub struct ClusteringSleuthInfo<W> {
    /// Join handle for the thread. Can be joined once the sleuth is done and dropped.
    pub thread: JoinHandle<io::Result<W>>,
    /// The `Sleuth` created - can be added as a layer for a tracing subscriber.
    pub sleuth: Sleuth<mpsc::SyncSender<SpanInfo>>,
}

/// Creates a new Sleuth that clusters spans and periodically outputs information about the clusters.
pub fn sleuth_with_clustering_thread<W>(
    mut parameters: ClusteringSleuthParameters<W>,
) -> ClusteringSleuthInfo<W>
where
    W: Write + Send + 'static,
{
    let (tx, rx) = mpsc::sync_channel(parameters.channel_limit);
    let thread = spawn(move || {
        let mut clustering = SleuthClassifier::new(parameters.clustering_parameters);
        let mut next_output = Instant::now() + parameters.output_period;
        while let Ok(span) = rx.recv() {
            clustering.learn_span(span);
            if next_output <= Instant::now() {
                output_clusters(&mut parameters.output, clustering.clusters_with_reset())?;
                next_output = Instant::now() + parameters.output_period;
            }
        }
        // Output the clusters one last time
        output_clusters(&mut parameters.output, clustering.clusters())?;
        Ok(parameters.output)
    });
    let sleuth = Sleuth::new(tx);
    ClusteringSleuthInfo { thread, sleuth }
}

/// Output the cluster information in a formatted manner to the output.
/// For now I'm hardcoding the format and info I'm outputting.
fn output_clusters<W>(
    output: &mut W,
    clusters: HashMap<&'static str, Vec<Cluster>>,
) -> io::Result<()>
where
    W: Write,
{
    for span_type in clusters.keys().sorted() {
        writeln!(output, "== {span_type} ==")?;
        for cluster in clusters.get(span_type).unwrap() {
            writeln!(
                output,
                "{} - {}/{}/{}/{}",
                cluster.name(),
                cluster.timing_micros.min(),
                cluster.timing_micros.value_at_percentile(0.5),
                cluster.timing_micros.value_at_quantile(0.99),
                cluster.timing_micros.max()
            )?;
        }
    }
    Ok(())
}

impl SpanInfoConsumer for mpsc::SyncSender<SpanInfo> {
    fn consume_span(&self, span_info: SpanInfo) {
        match self.try_send(span_info) {
            Ok(()) => (),
            Err(TrySendError::Full(_)) => {
                // Intentionally ignore errors if the channel is full.
                // We're favoring losing some diagnostic data over blocking the main program
            }
            r => r.unwrap(),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        io::Cursor,
        sync::{Arc, Mutex},
    };

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

    #[test]
    fn clustering_thread() {
        let output = Cursor::new(Vec::new());
        let parameters = ClusteringSleuthParameters::new(output);
        let sleuth_info = sleuth_with_clustering_thread(parameters);
        {
            let subscriber = Registry::default().with(sleuth_info.sleuth);
            tracing::subscriber::with_default(subscriber, || {
                for _ in 0..100 {
                    work();
                }
            });
        }
        let output = sleuth_info.thread.join().unwrap().unwrap().into_inner();
        let output = String::from_utf8(output).unwrap();
        assert!(output.contains("== work =="), "Invalid output: {output}");
        assert!(
            output.contains("(sub_work: 0 - 0)"),
            "Invalid output: {output}"
        );
    }
}
