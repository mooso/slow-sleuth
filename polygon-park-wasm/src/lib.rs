use std::sync::{Arc, Mutex};

use polygon_park::PolygonPark;
use rand::thread_rng;
use slow_sleuth::{clustering::SleuthClassifier, Sleuth, SpanInfo, SpanInfoConsumer};
use tracing_subscriber::{prelude::*, Registry};
use wasm_bindgen::prelude::wasm_bindgen;

struct LockedClassifier {
    classifier: Arc<Mutex<SleuthClassifier>>,
}

impl SpanInfoConsumer for LockedClassifier {
    fn consume_span(&self, span_info: SpanInfo) {
        self.classifier.lock().unwrap().learn_span(span_info);
    }
}

#[wasm_bindgen]
pub struct WasmPark {
    park: PolygonPark,
    classifier: Arc<Mutex<SleuthClassifier>>,
}

#[wasm_bindgen]
impl WasmPark {
    pub fn new(width: f32, height: f32) -> WasmPark {
        let mut rng = thread_rng();
        let park = PolygonPark::new_random(&mut rng, width, height);
        let classifier = Arc::new(Mutex::new(SleuthClassifier::default()));
        let sleuth = Sleuth::new(LockedClassifier {
            classifier: classifier.clone(),
        });
        let subscriber = Registry::default().with(sleuth);
        tracing::subscriber::set_global_default(subscriber).unwrap();
        WasmPark { park, classifier }
    }

    pub fn tick(&mut self, millis_elapsed: f32) {
        self.park.tick(millis_elapsed);
    }

    pub fn num_polygons(&self) -> usize {
        self.park.polygons.len()
    }

    pub fn polygon_color(&self, polygon_index: usize) -> u32 {
        self.park.polygons[polygon_index].color
    }

    pub fn num_vertices(&self, polygon_index: usize) -> usize {
        self.park.polygons[polygon_index].geometry.vertices.len()
    }

    pub fn vertix_x(&self, polygon_index: usize, point_index: usize) -> f32 {
        self.park.polygons[polygon_index].geometry.vertices[point_index].x
    }

    pub fn vertix_y(&self, polygon_index: usize, point_index: usize) -> f32 {
        self.park.polygons[polygon_index].geometry.vertices[point_index].y
    }

    pub fn num_tick_clusters(&self) -> usize {
        self.classifier
            .lock()
            .unwrap()
            .clusters()
            .get("tick")
            .map(Vec::len)
            .unwrap_or_default()
    }

    pub fn tick_cluster_label_range(&self, cluster_index: usize, label: &str) -> String {
        self.classifier
            .lock()
            .unwrap()
            .clusters()
            .get("tick")
            .unwrap()[cluster_index]
            .bins
            .get(label)
            .map(|b| format!("{b}"))
            .unwrap_or_default()
    }

    pub fn tick_cluster_micros_quantile(&self, cluster_index: usize, quantile: f64) -> u64 {
        self.classifier
            .lock()
            .unwrap()
            .clusters()
            .get("tick")
            .unwrap()[cluster_index]
            .timing_micros
            .value_at_quantile(quantile)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basics() {
        let mut park = WasmPark::new(100., 100.);
        park.tick(10.);
        assert!(park.num_polygons() > 0);
        assert!(park.num_vertices(0) > 0);
        assert!(park.vertix_x(0, 0) >= 0. && park.vertix_x(0, 0) <= 100.);
        assert!(park.vertix_y(0, 0) >= 0. && park.vertix_y(0, 0) <= 100.);
    }
}
