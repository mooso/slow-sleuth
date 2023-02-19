use polygon_park::PolygonPark;
use rand::thread_rng;
use wasm_bindgen::prelude::wasm_bindgen;

#[wasm_bindgen]
pub struct WasmPark {
    park: PolygonPark,
}

#[wasm_bindgen]
impl WasmPark {
    pub fn new(width: f32, height: f32) -> WasmPark {
        let mut rng = thread_rng();
        let park = PolygonPark::new_random(&mut rng, width, height);
        WasmPark { park }
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
