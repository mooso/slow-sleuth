//! Toy library for simulating moving polygons that can collide with each other.

use std::ops;

use rand::Rng;
use tracing::instrument;

/// A point or a vector in the 2D space.
#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub struct Point {
    pub x: f32,
    pub y: f32,
}

impl ops::Sub for Point {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Point {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
        }
    }
}


impl ops::Add for Point {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Point {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
        }
    }
}
impl ops::Neg for Point {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Point {
            x: -self.x,
            y: -self.y,
        }
    }
}

impl ops::Mul<f32> for Point {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        Point {
            x: self.x * rhs,
            y: self.y * rhs,
        }
    }
}

impl ops::Div<f32> for Point {
    type Output = Self;

    fn div(self, rhs: f32) -> Self::Output {
        Point {
            x: self.x / rhs,
            y: self.y / rhs,
        }
    }
}

impl Point {
    /// Dot product of myself and another 2D vector
    pub fn dot(&self, rhs: &Self) -> f32 {
        self.x * rhs.x + self.y * rhs.y
    }

    /// The normal (perpendicular) of this 2D vector, *not* normalized to unit length.
    pub fn normal(&self) -> Self {
        Self {
            x: -self.y,
            y: self.x,
        }
    }

    /// The magnitude of this 2D vector
    pub fn magnitude(&self) -> f32 {
        (self.x.powi(2) + self.y.powi(2)).sqrt()
    }

    /// Normalize this 2D vector to unit length
    pub fn normalize(&self) -> Self {
        *self / self.magnitude()
    }
}

/// A 2D polygon
#[derive(Clone, PartialEq, Debug)]
pub struct Polygon {
    pub vertices: Vec<Point>,
}

/// An iterator over the edges of a polygon.
pub struct Edges<'a> {
    vertices: &'a Vec<Point>,
    current_index: usize,
}

impl Iterator for Edges<'_> {
    type Item = Point;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index == self.vertices.len() {
            return None;
        }
        let p1 = self.vertices.get(self.current_index).unwrap();
        let p2 = self
            .vertices
            .get((self.current_index + 1) % self.vertices.len())
            .unwrap();
        self.current_index += 1;
        Some(*p1 - *p2)
    }
}

/// A projection of a shape onto an axis.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Projection {
    /// The minimum of the projection.
    pub min: f32,
    /// The maximum of the projection.
    pub max: f32,
}

impl Projection {
    /// Checks if myself and the given projection overlap. If we do, return the overlap projection.
    pub fn overlap(&self, rhs: &Self) -> Option<Projection> {
        let projection = Projection {
            min: self.min.max(rhs.min),
            max: self.max.min(rhs.max),
        };
        if projection.min <= projection.max {
            Some(projection)
        } else {
            None
        }
    }
}

impl Polygon {
    /// Iterate over the edges of the polygon.
    pub fn edges(&self) -> Edges<'_> {
        Edges {
            vertices: &self.vertices,
            current_index: 0,
        }
    }

    /// Project the polygon onto a given axis.
    // TODO I commented out the instument attribute for testing
    //#[instrument]
    pub fn project(&self, axis: Point) -> Projection {
        let mut min = axis.dot(self.vertices.first().unwrap());
        let mut max = min;
        for p in self.vertices.iter().skip(1) {
            let p = axis.dot(p);
            if p < min {
                min = p;
            } else if p > max {
                max = p;
            }
        }
        Projection { min, max }
    }

    /// Check if this polygon collides with another given polygon. Returns the minimum translation vector if a collision exists.
    pub fn get_mtv(&self, other: &Self) -> Option<Point> {
        // Check using an SAT (Separating Axis Theorem) algorithm.
        // See e.g. https://dyn4j.org/2010/01/sat/ for a description.

        let mut min_distance = f32::MAX;
        let mut mtv_direction = Point { x: 0., y: 0. };

        for edge in self.edges().chain(other.edges()) {
            let axis = edge.normal().normalize();
            let p1 = self.project(axis);
            let p2 = other.project(axis);
            let overlap = p1.overlap(&p2);
            if overlap.is_none() {
                // If we find one axis in which the projections of the shapes don't overlap, they're not in collision
                return None;
            }

            let overlap_length = overlap.unwrap().max - overlap.unwrap().min;
            if overlap_length < min_distance {
                min_distance = overlap_length;
                mtv_direction = axis;
            }
        }
        return Some(mtv_direction * min_distance);
    }

    /// Check if this polygon collides with another given polygon. Returns false if not colliding.
    pub fn check_collision(&self, other: &Self) -> bool {
        return self.get_mtv(other).is_some();
    }

}

/// A polygon in the park, that can move around.
#[derive(Clone, PartialEq, Debug)]
pub struct MovingPolygon {
    /// The geometry and current location of the polygon.
    pub geometry: Polygon,
    /// The maass of the polygon (for better looking collisions).
    pub mass: f32,
    /// The color of the polygon.
    pub color: u32,
    /// The current velocity of the polygon, in units/second.
    pub velocity: Point,
}

/// A polygon park! Full of moving exciting polygon.
#[derive(Clone, PartialEq, Debug)]
pub struct PolygonPark {
    /// The polygons in the park.
    pub polygons: Vec<MovingPolygon>,
    /// The width of the park (x positions should all be confined to `0-width`).
    pub width: f32,
    /// The height of the park (y positions should all be confined to `0-height`).
    pub height: f32,
}

fn signum(input: f32) -> f32 {
    if input < 0. {
        return -1.;
    }
    else if input == 0. {
        return 0. ;
    }
    else {
        return 1.;
    }
}

/// Sets sign of input to sign of sign
fn ensure_sign(input: &mut f32, sign: f32) {
    if signum(*input) != signum(sign) {
        *input *= -1.;
    }
}

fn translate_polygon(polygon: &mut Polygon, translation: Point) {
    for vertex in polygon.vertices.iter_mut() {
        *vertex = *vertex + translation;
    }
}

/// Resolves collision specified by mtv by moving penetrating_polygon outside of second_polygon
fn resolve_collision(penetrating_polygon: &mut Polygon, second_polygon: Polygon, mtv: Point) {
    let difference = penetrating_polygon.vertices[0] - second_polygon.vertices[0];
    translate_polygon(penetrating_polygon, mtv * signum(difference.dot(&mtv)) * 0.95);
}

fn generate_square<R: Rng>(_rng: &mut R, top_left_position: Point, length: f32) -> Polygon {

    let square = Polygon {
        vertices: vec![
            top_left_position,
            top_left_position + Point { x: length, y: 0. },
            top_left_position + Point { x: length, y: length },
            top_left_position + Point { x: 0., y: length },

          ],
      };

    return square;
}

fn generate_polygon<R: Rng>(_rng: &mut R, center_position: Point, radius: f32) -> Polygon {

    const NUM_VERTICES: u32 = 64;
    const STEP_RADIANS: f32 = 2. * std::f32::consts::PI / NUM_VERTICES as f32;
    const TERMINATION_PROBABILITY: f32 = 0.9;

    let mut success = false;
    let mut verts: Vec<Point> = Vec::new();

    // generate NUM_VERTICES placed on a circle. Randomly delete some vertices to create unique convex shapes .

    while !success {
        verts.clear();
        for i in 0..NUM_VERTICES {
            if _rng.gen_range(0f32..1f32) > TERMINATION_PROBABILITY {
                let vert = center_position + Point { x: f32::cos(i as f32 * STEP_RADIANS), y: f32::sin(i as f32 * STEP_RADIANS) } * radius;
                verts.push(vert);
            }
        }

        success = verts.len() >= 3;
    }

    return Polygon { vertices: verts };
}

impl PolygonPark {
    /// Generate a new random park (full of exciting random polygons) of the given width/height.
    pub fn new_random<R: Rng>(_rng: &mut R, width: f32, height: f32) -> PolygonPark {
        const NUM_POLYGONS: u32 = 16;
        const POSSIBLE_VELOCITY: std::ops::Range<f32> = std::ops::Range { start: -100., end: 100. };
        const POSSIBLE_RADIUS: std::ops::Range<f32> = std::ops::Range { start: 10., end: 50. };

        let mut polygons: Vec<MovingPolygon> = Vec::new();

        for _ in 0..NUM_POLYGONS {
            let polygon_radius = _rng.gen_range(POSSIBLE_RADIUS);
            let polygon_center = Point { x: _rng.gen_range(0f32..width - polygon_radius), y: _rng.gen_range(0f32..height - polygon_radius) };

            let polygon = MovingPolygon {
                geometry: generate_polygon(_rng, polygon_center, polygon_radius),
                mass: polygon_radius / 2.,
                color: _rng.gen_range(u32::MIN..u32::MAX),
                velocity: Point { x: _rng.gen_range(POSSIBLE_VELOCITY), y: _rng.gen_range(POSSIBLE_VELOCITY) },
            };

            polygons.push(polygon);
        }

        PolygonPark {
            polygons: polygons,
            width,
            height,
        }
    }


    /// Tick the park - simulate the movement by advancing time by the given number of milli-seconds.
    // TODO I commented out the instrument attribute for testing
    //#[instrument]
    pub fn tick(&mut self, millis_elapsed: f32) {

        for i in 0..self.polygons.len()
        {

            // collision between polygons and other polygons
            for j in (i + 1)..self.polygons.len()
            {
                let collision_mtv = self.polygons[i].geometry.get_mtv(&self.polygons[j].geometry);
                if collision_mtv.is_some()
                {
                    let second_polygon = self.polygons[j].geometry.clone();
                    resolve_collision(&mut self.polygons[i].geometry, second_polygon, collision_mtv.unwrap());

                    let mass_ratio = self.polygons[j].mass / self.polygons[i].mass;

                    let temp = self.polygons[i].velocity;
                    self.polygons[i].velocity = self.polygons[j].velocity * mass_ratio;
                    self.polygons[j].velocity = temp * 1. / mass_ratio;
                }
            }

            // translation
            {
                let polygon = &mut self.polygons[i];
                translate_polygon(&mut polygon.geometry, polygon.velocity * millis_elapsed / 1000.);
            }


            // collisions between polygons and park edges
            for j in 0..self.polygons[i].geometry.vertices.len() {
                let vertex = self.polygons[i].geometry.vertices[j];

                if vertex.x < 0. || vertex.x > self.width { // colliding with left or right edge
                    ensure_sign(&mut self.polygons[i].velocity.x, -vertex.x);
                    translate_polygon(&mut self.polygons[i].geometry, Point{ x: 1. * signum(-vertex.x), y: 0. });
                }

                if vertex.y < 0. || vertex.y > self.height { // colliding with bottom or top edge
                    ensure_sign(&mut self.polygons[i].velocity.y, -vertex.y);
                    translate_polygon(&mut self.polygons[i].geometry, Point{ x: 0., y: 1. * signum(-vertex.y) });
                }
            }
        }

    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn colliding() {
        let p1 = Polygon {
            vertices: vec![
                Point { x: 0., y: 0. },
                Point { x: 10., y: 0. },
                Point { x: 10., y: 10. },
                Point { x: 0., y: 10. },
            ],
        };
        let p2 = Polygon {
            vertices: vec![
                Point { x: 9., y: 5. },
                Point { x: 19., y: 5. },
                Point { x: 19., y: 15. },
                Point { x: 9., y: 15. },
            ],
        };
        assert!(p2.check_collision(&p1));
        assert!(p1.check_collision(&p2));
        assert!(p2.check_collision(&p2));
        assert!(p1.check_collision(&p1));
    }

    #[test]
    fn far_apart() {
        let p1 = Polygon {
            vertices: vec![
                Point { x: 0., y: 0. },
                Point { x: 10., y: 0. },
                Point { x: 10., y: 10. },
                Point { x: 0., y: 10. },
            ],
        };
        let p2 = Polygon {
            vertices: vec![
                Point { x: 25., y: 5. },
                Point { x: 35., y: 5. },
                Point { x: 35., y: 15. },
                Point { x: 25., y: 15. },
            ],
        };
        assert!(!p1.check_collision(&p2));
        assert!(!p2.check_collision(&p1));
    }
}
