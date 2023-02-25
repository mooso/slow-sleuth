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

    /// Check if this polygon collides with another given polygon. Returns false if not colliding.
    pub fn check_collision(&self, other: &Self) -> bool {
        // Check using an SAT (Separating Axis Theorem) algorithm.
        // See e.g. https://dyn4j.org/2010/01/sat/ for a description.
        for edge in self.edges().chain(other.edges()) {
            let axis = edge.normal();
            let p1 = self.project(axis);
            let p2 = other.project(axis);
            if p1.overlap(&p2).is_none() {
                // If we find one axis in which the projections of the shapes don't overlap, they're not in collision
                return false;
            }
        }
        true
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

/// Returns center of polygon
fn average_point(polygon: &Polygon) -> Point {
    let mut accumulator = Point{ x: 0., y: 0. };

    for vertex in &polygon.vertices {
        accumulator = accumulator + *vertex;
    }

    return accumulator / polygon.vertices.len() as f32;
}

/// Return in_vector reflected about normal_vector. Expects normal_vector to be normalized
fn reflect_vector(in_vector: &Point, normal_vector: &Point) -> Point {
    return *in_vector - (*normal_vector * 2. * normal_vector.dot(in_vector));
}

/// Linearly interpolates between a and b according to alpha. expects 0 <= alpha <= 1
fn lerp<T: std::ops::Mul<f32, Output = T> + std::ops::Add<T, Output = T> >(a: T, b: T, alpha: f32) -> T {
    return a * alpha + b * (1. - alpha);
}

impl PolygonPark {
    /// Generate a new random park (full of exciting random polygons) of the given width/height.
    pub fn new_random<R: Rng>(_rng: &mut R, width: f32, height: f32) -> PolygonPark {
        // TODO: Actually randomly generate

    const NUM_POLYGONS: u32 = 16;

    const POSSIBLE_SQUARE_LENGTH: std::ops::Range<f32> = std::ops::Range { start: 10., end: 50. };

    const POSSIBLE_VELOCITY: std::ops::Range<f32> = std::ops::Range { start: -100., end: 100. };

	let mut squares: Vec<MovingPolygon> = Vec::new();

	for _ in 0..NUM_POLYGONS {

		let square_length = _rng.gen_range(POSSIBLE_SQUARE_LENGTH);

		let square_top_left = Point { x: _rng.gen_range(0f32..width - square_length), y: _rng.gen_range(0f32..height - square_length) };

		let square = Polygon {
			vertices: vec![
				square_top_left,
				square_top_left + Point { x: square_length, y: 0. },
				square_top_left + Point { x: square_length, y: square_length },
				square_top_left + Point { x: 0., y: square_length },

		      ],
		  };

		let square = MovingPolygon {
		    geometry: square,
		    mass: square_length / 2.,
		    //color: 0x00DD00,
                    color: _rng.gen_range(u32::MIN..u32::MAX),
		    velocity: Point { x: _rng.gen_range(POSSIBLE_VELOCITY), y: _rng.gen_range(POSSIBLE_VELOCITY) },
		};

		squares.push(square);
	}

        PolygonPark {
            polygons: squares,
            width,
            height,
        }
    }


    /// Tick the park - simulate the movement by advancing time by the given number of milli-seconds.
    // TODO I commented out the instrument attribute for testing
    //#[instrument]
    pub fn tick(&mut self, millis_elapsed: f32) {
        // TODO: Actually simulate

        for i in 0..self.polygons.len()
        {

            // collision between polygons and other polygons
            for j in (i + 1)..self.polygons.len()
            {
                if self.polygons[i].geometry.check_collision(&self.polygons[j].geometry)
                {
                    self.polygons[i].color = 0xFFFF0000;
                    self.polygons[j].color = 0xFFFF0000;

                    let polygon_center = average_point(&self.polygons[i].geometry);
                    let other_polygon_center = average_point(&self.polygons[j].geometry);

                    let difference = other_polygon_center - polygon_center;
                    translate_polygon(&mut self.polygons[i].geometry, -difference.normalize() * 1.);
                    translate_polygon(&mut self.polygons[j].geometry, difference.normalize() * 1.);

                    let old_velocity0 = self.polygons[i].velocity.clone();
                    let old_velocity1 = self.polygons[j].velocity.clone();

                    let mass_ratio = self.polygons[j].mass / self.polygons[i].mass;

                    //self.polygons[i].velocity = old_velocity1 * mass_ratio;
                    //self.polygons[j].velocity = old_velocity0 * 1. / mass_ratio; 

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
