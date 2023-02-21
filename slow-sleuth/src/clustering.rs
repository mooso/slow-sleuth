//! Clustering for span information to cluster together similar-work spans and have timing statistics per cluster.

use std::{collections::HashMap, fmt::Display};

use hdrhistogram::Histogram;
use itertools::Itertools;

use crate::SpanInfo;

/// The parameters for the clustering algorithm.
pub struct ClusteringParameters {
    /// The maximum points per span type to use just for learning the cluster shapes.
    pub max_points_to_learn: usize,
    /// The target number of clusters per span type.
    pub target_clusters_per_span_type: usize,
}

impl Default for ClusteringParameters {
    fn default() -> Self {
        Self {
            max_points_to_learn: 10,
            target_clusters_per_span_type: 4,
        }
    }
}

/// A learner for clustering - doesn't collect timing info, just learns the shapes of the clusters.
#[derive(Default)]
struct ClusterLearner {
    /// Labels for the vector of the classifier - so if the span typically goes through steps X, Y & Z, this
    /// vector would be `["X", "Y", "Z"]`
    vector_labels: Vec<&'static str>,
    /// The spans encountered so far, represented as vectors. So if a span did step X 12 times, Y 13 times and Z
    /// 14 times, and the `vector_labels` was `["X", "Y", "Z"]`, then it would be represented as `[12, 13, 14]`.
    points: Vec<Vec<usize>>,
}

impl ClusterLearner {
    /// Take the given span and learn from it.
    pub fn learn_span(&mut self, mut span: SpanInfo) {
        // Represent the span as a vector
        let mut point = Vec::with_capacity(self.vector_labels.len());
        // First put in the points for the steps we know of.
        for label in self.vector_labels.iter() {
            point.push(span.child_counts.remove(label).unwrap_or_default());
        }
        // Then learn any new steps we haven't seen before.
        for (label, count) in span.child_counts {
            self.vector_labels.push(label);
            point.push(count);
        }
        // Add the span as a vector.
        self.points.push(point);
    }

    /// Check if we're done learning and ready to turn into a classifier.
    pub fn done(&self, parameters: &ClusteringParameters) -> bool {
        self.points.len() >= parameters.max_points_to_learn
    }

    /// Turn this into a classifier.
    pub fn into_classifier(self, parameters: &ClusteringParameters) -> Classifier {
        // Find the minimum/maximum for each count of steps as vectors.
        let mut minima = vec![usize::MAX; self.vector_labels.len()];
        let mut maxima = vec![0; self.vector_labels.len()];
        for point in self.points.iter() {
            for (i, &v) in point.iter().enumerate() {
                minima[i] = minima[i].min(v);
                maxima[i] = maxima[i].max(v);
            }
        }
        // The maxima will be used to normalize the vectors, so that a vector with maximum
        // seen so far would have magnitude (norm) of 1.0. Since we're targeting X clusters,
        // define the bin width as dividing the spread between the minimum seen and maximum
        // by the target X.
        // Note that we may still get more than X clusters, if we see more than maximum or
        // less than minimum. We could bound this by e.g. putting everything less than minimum
        // into a bin and everything more than maximum into a bin. We should. Future improvement.
        let spread = 1.0 - vector_norm(&minima, &maxima);
        let bin_width = spread / (parameters.target_clusters_per_span_type as f64);
        Classifier {
            vector_labels: self.vector_labels,
            bin_width,
            maxima,
            clusters: Default::default(),
        }
    }
}

/// Get the norm/magnitude of a vector of step execution counts based on the maxima of each count seen.
fn vector_norm(vector: &[usize], maxima: &[usize]) -> f64 {
    vector
        .iter()
        .zip(maxima.iter())
        .map(|(&v, &m)| ((v as f64) / (m as f64)).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// The range of counts for a given step in a cluster.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct BinCountRange {
    pub low: usize,
    pub high: usize,
}

impl BinCountRange {
    /// Construct the range from the label of the bin (which bin from 0 - N of N bins) given
    /// the bin width and the max expected count.
    /// For example - if this is the second bin and the bin width is 0.25, and we expected the
    /// maximum to be 100, then the range for this bin should be 25 - 50.
    fn from_bin_label(bin_label: usize, maximum: usize, bin_width: f64) -> BinCountRange {
        let low = ((bin_label as f64) * bin_width * (maximum as f64)).round() as usize;
        let high = (((bin_label + 1) as f64) * bin_width * (maximum as f64)).round() as usize;
        BinCountRange { low, high }
    }
}

impl Display for BinCountRange {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} - {}", self.low, self.high)
    }
}

/// A cluster of span executions and a histogram of the timings observed.
#[derive(Clone)]
pub struct Cluster {
    /// The ranges of expected counts for the steps in the cluster.
    /// This is purely for humans to make sense of what this cluster represents.
    pub bins: HashMap<&'static str, BinCountRange>,
    /// The histogram of timings (as micro-seconds).
    pub timing_micros: Histogram<u64>,
}

impl Cluster {
    /// A human-readable name for the cluster.
    pub fn name(&self) -> String {
        self.bins
            .iter()
            .sorted_by_key(|(&label, _)| label)
            .map(|(&label, &bin)| format!("({label}: {bin})"))
            .join(",")
    }
}

/// A learned classifier for a span type.
struct Classifier {
    /// Labels for the vector of the classifier - so if the span typically goes through steps X, Y & Z, this
    /// vector would be `["X", "Y", "Z"]`
    vector_labels: Vec<&'static str>,
    /// The width of each bin as the normalized magnitude (should be in the 0-1 range).
    bin_width: f64,
    /// The maxima observed during the learning phase for each step count.
    maxima: Vec<usize>,
    /// Clusters observed so far.
    clusters: HashMap<usize, Cluster>,
}

impl Classifier {
    /// Take the given span, find its cluster and record the timing there.
    pub fn learn_span(&mut self, span: SpanInfo) {
        let vector = self.to_vector(&span.child_counts);
        let bin_label = self.bin_label(vector_norm(&vector, &self.maxima));
        let cluster = self.clusters.entry(bin_label).or_insert_with(|| Cluster {
            bins: Self::bins_for_cluster(
                bin_label,
                &self.maxima,
                &self.vector_labels,
                self.bin_width,
            ),
            timing_micros: Histogram::new_with_bounds(1, u64::MAX, 3).unwrap(),
        });
        cluster
            .timing_micros
            .record(span.duration.as_micros() as u64)
            .unwrap();
    }

    /// Represent the given span as a vector.
    fn to_vector(&self, child_counts: &HashMap<&'static str, usize>) -> Vec<usize> {
        self.vector_labels
            .iter()
            .map(|&label| child_counts.get(label).copied().unwrap_or_default())
            .collect()
    }

    /// Get the label (an integer) for a given vector.
    fn bin_label(&self, vector_norm: f64) -> usize {
        (vector_norm / self.bin_width).round() as usize
    }

    /// Construct the bins for a cluster.
    fn bins_for_cluster(
        bin_label: usize,
        maxima: &[usize],
        vector_labels: &[&'static str],
        bin_width: f64,
    ) -> HashMap<&'static str, BinCountRange> {
        vector_labels
            .iter()
            .zip(maxima.iter())
            .map(|(&label, &maximum)| {
                (
                    label,
                    BinCountRange::from_bin_label(bin_label, maximum, bin_width),
                )
            })
            .collect()
    }
}

/// A per-span-type learner of executions of spans.
enum PerSpanTypeLearner {
    /// A learner in the learning phase, still figuring out the shapes of the clusters.
    ClusterLearner(ClusterLearner),
    /// A learner in the clustering phase, putting span executions into clusters and recording times.
    Classifier(Classifier),
}

impl PerSpanTypeLearner {
    /// Evolve the learner from learning to clustering (does nothing if already in clustering).
    pub fn evolve(self, parameters: &ClusteringParameters) -> Self {
        match self {
            PerSpanTypeLearner::ClusterLearner(learner) => {
                PerSpanTypeLearner::Classifier(learner.into_classifier(parameters))
            }
            PerSpanTypeLearner::Classifier(_) => self,
        }
    }
}

/// A clustering classifier that can learn clusters of span executions and timings for them.
#[derive(Default)]
pub struct SleuthClassifier {
    /// Per-span-type learner.
    per_span_learner: HashMap<&'static str, PerSpanTypeLearner>,
    /// Parameters for the algorithm.
    parameters: ClusteringParameters,
}

impl SleuthClassifier {
    /// Create a new classifier with the given clustering parameters.
    pub fn new(parameters: ClusteringParameters) -> Self {
        Self {
            per_span_learner: Default::default(),
            parameters,
        }
    }

    /// Learn the execution of the given span.
    pub fn learn_span(&mut self, span: SpanInfo) {
        let span_name = span.span_name;
        // Get the learner for this span type, or start it off as a cluster learner.
        let learner = self
            .per_span_learner
            .entry(span_name)
            .or_insert(PerSpanTypeLearner::ClusterLearner(Default::default()));
        // Learn the execution, and figure out if it should evolve.
        let should_evolve = match learner {
            PerSpanTypeLearner::ClusterLearner(learner) => {
                learner.learn_span(span);
                learner.done(&self.parameters)
            }
            PerSpanTypeLearner::Classifier(learner) => {
                learner.learn_span(span);
                false
            }
        };
        if should_evolve {
            // Need to evolve it into a clustering learner. Remove and replace.
            let learner = self.per_span_learner.remove(span_name).unwrap();
            self.per_span_learner
                .insert(span_name, learner.evolve(&self.parameters));
        }
    }

    /// Get the clusters learned so far.
    pub fn clusters(&self) -> HashMap<&'static str, Vec<Cluster>> {
        self.per_span_learner
            .iter()
            .map(|(&name, learner)| {
                let clusters = match learner {
                    PerSpanTypeLearner::ClusterLearner(_) => vec![],
                    PerSpanTypeLearner::Classifier(classifier) => {
                        classifier.clusters.values().cloned().collect()
                    }
                };
                (name, clusters)
            })
            .collect()
    }

    /// Get the clusters learned so far, and reset the timings after getting them.
    /// Useful for periodically getting the timings then starting a fresh period.
    pub fn clusters_with_reset(&mut self) -> HashMap<&'static str, Vec<Cluster>> {
        self.per_span_learner
            .iter_mut()
            .map(|(&name, learner)| {
                let clusters = match learner {
                    PerSpanTypeLearner::ClusterLearner(_) => vec![],
                    PerSpanTypeLearner::Classifier(classifier) => classifier
                        .clusters
                        .iter_mut()
                        .map(|(_, cluster)| {
                            let clone = cluster.clone();
                            cluster.timing_micros.reset();
                            clone
                        })
                        .collect(),
                };
                (name, clusters)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use std::time::Duration;

    use super::*;

    /// A string representation for a Vec of clusters - used to understand and test.
    fn to_string(clusters: &Vec<Cluster>) -> String {
        clusters
            .iter()
            .map(|c| {
                format!(
                    "[{}: ({}, {}, {})]",
                    c.name(),
                    c.timing_micros.min(),
                    c.timing_micros.mean(),
                    c.timing_micros.max()
                )
            })
            .sorted()
            .join(",")
    }

    #[test]
    pub fn linear_span() {
        let mut classifier = SleuthClassifier::new(ClusteringParameters {
            max_points_to_learn: 10,
            target_clusters_per_span_type: 4,
        });
        for _ in 0..100 {
            for j in 0..10 {
                let mut child_counts = HashMap::new();
                child_counts.insert("step", j + 1);
                classifier.learn_span(SpanInfo {
                    span_name: "work",
                    child_counts,
                    duration: Duration::from_micros((j + 1) as u64),
                })
            }
        }
        let clusters = classifier.clusters();
        let my_clusters = clusters.get("work").unwrap();
        assert_eq!(
            "[(step: 0 - 2): (1, 1, 1)],[(step: 2 - 5): (2, 2.5, 3)],[(step: 5 - 7): (4, 4.5, 5)],[(step: 7 - 9): (6, 6.5, 7)],[(step: 9 - 11): (8, 9, 10)]",
            to_string(my_clusters)
        );
    }
}
