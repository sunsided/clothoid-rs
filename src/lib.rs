pub mod fit;
pub mod optimizer;

/// The square root of π.
const PI_SQRT: f64 = 1.7724538509055160272981674833411451827975494561223871282138077898f64;

/// The inverse of the square root of π, i.e. `1.0 / PI_SQRT` (see [`PI_SQRT`]).
const INV_PI_SQRT: f64 = 0.5641895835477562869480794515607725858440506293289988568440857217f64;

#[derive(Debug, Default, Copy, Clone, PartialOrd, PartialEq)]
pub struct Point2 {
    pub x: f64,
    pub y: f64,
}

pub struct Clothoid {
    /// The scaling factor of the clothoid.
    a: f64,
}

impl Clothoid {
    pub fn new(a: f64) -> Self {
        Self { a }
    }

    #[inline]
    pub fn direction_angle(&self, arc_length: f64) -> f64 {
        0.5 * (arc_length * arc_length) / (self.a * self.a)
    }

    #[inline(always)]
    fn calculate(&self, t: f64) -> Point2 {
        #[cfg(feature = "fresnel")]
        {
            self.calculate_fresnl(t)
        }

        #[cfg(not(feature = "fresnel"))]
        {
            self.calculate_approx(t)
        }
    }

    #[cfg(feature = "fresnel")]
    fn calculate_fresnl(&self, t: f64) -> Point2 {
        let (s, c) = fresnel::fresnl(t * INV_PI_SQRT);
        Point2 {
            x: self.a * PI_SQRT * s,
            y: self.a * PI_SQRT * c,
        }
    }

    fn calculate_approx(&self, t: f64) -> Point2 {
        let fsc = FresnelSinCos::compute(t * INV_PI_SQRT);
        Point2 {
            x: self.a * PI_SQRT * fsc.sin,
            y: self.a * PI_SQRT * fsc.cos,
        }
    }
}

/// Auxiliary functions `f(x)` and `g(x)` for calculating the
/// Fresnel sines and cosines, `S(x)` and `C(x)`.
struct AuxFg {
    pub f: f64,
    pub g: f64,
}

impl AuxFg {
    /// Calculates the auxiliary functions `f` and `g` for the Fresnel integrals.
    ///
    /// ## Sources
    /// Doran K. Wilde, "Computing Clothoid Segments for Trajectory Generation".
    /// IEEE/RSJ International Conference on Intelligent Robots and Systems, October 2009.
    ///
    /// Abramowitz, Milton and Stegun, Irene A., (Editors), "Handbook of Mathematical
    /// Functions with Formulas, Graphs, and Mathematical Tables".
    /// National Bureau of Standards Applied Mathematics Series, No. 55, June 1964, pp. 295-303.
    pub fn compute(x: f64) -> Self {
        let x2 = x * x;
        let x3 = x * x * x;

        let f = (1. + 0.926 * x) / (2. + 1.792 * x + 3.104 * x2);
        let g = 1. / (2. + 4.142 * x + 3.492 * x2 + 6.670 * x3);

        Self { f, g }
    }
}

/// Fresnel sines and cosines, S(x) and C(x).
struct FresnelSinCos {
    /// C(x)
    pub cos: f64,
    /// S(x)
    pub sin: f64,
}

impl FresnelSinCos {
    /// Calculates the Fresnel sines and cosines, `S(x)` and `C(x)`
    /// by means of the auxiliary functions f(x)` and `g(x)` (see [`AuxFg::compute`]).
    pub fn compute(x: f64) -> Self {
        let aux = AuxFg::compute(x);
        let (sin, cos) = (x * x * std::f64::consts::FRAC_PI_2).sin_cos();
        Self {
            cos: 0.5 + aux.f * sin - aux.g * cos,
            sin: 0.5 - aux.f * cos - aux.g * sin,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_float_eq::*;

    #[test]
    fn it_works() {
        let clothoid = Clothoid::new(1.);
        let alpha = clothoid.direction_angle(0.);
        assert_eq!(alpha, 0.);
    }

    #[test]
    fn calculate() {
        let clothoid = Clothoid::new(8.);
        let pt = clothoid.calculate(0.);
        assert_f64_near!(pt.x, 0.);
        assert_f64_near!(pt.y, 0.);

        let pt = clothoid.calculate_approx(std::f64::consts::PI);
        assert!((pt.x - 6.77).abs() < 0.01);
        assert!((pt.y - 4.59).abs() < 0.01);
    }

    #[test]
    #[cfg(feature = "fresnel")]
    fn calculate_fresnl_works() {
        let clothoid = Clothoid::new(8.);
        let pt = clothoid.calculate_fresnl(0.);
        assert_f64_near!(pt.x, 0.);
        assert_f64_near!(pt.y, 0.);

        let pt = clothoid.calculate_fresnl(std::f64::consts::PI);
        assert_f64_near!(pt.x, 6.7669799976205);
        assert_f64_near!(pt.y, 4.615663254508842);

        // http://jsxgraph.uni-bayreuth.de/wiki/index.php/Euler's_spiral_(Clothoid)
    }

    #[test]
    fn calculate_approx_works() {
        let clothoid = Clothoid::new(8.);
        let pt = clothoid.calculate_approx(0.);
        assert_f64_near!(pt.x, 0.);
        assert_f64_near!(pt.y, 0.);

        let pt = clothoid.calculate_approx(std::f64::consts::PI);
        assert_f64_near!(pt.x, 6.777113091819308);
        assert_f64_near!(pt.y, 4.588251163366395);
    }

    #[test]
    fn pochhammer() {
        // https://dlmf.nist.gov/7.12
        // https://dlmf.nist.gov/5.2#iii

        fn p(a: f64, n: usize) -> f64 {
            if n == 0 {
                return 1.;
            }
            let mut product = 1.;
            for i in 0..n {
                product *= a + (i as f64)
            }
            product
        }

        assert_eq!(p(0.5, 0), 1.);
        assert_eq!(p(0.5, 1), 0.5);
        assert_eq!(p(0.5, 2), 0.75);
        assert_eq!(p(0.5, 3), 1.875);
        assert_eq!(p(0.5, 7), 1055.7421875);

        // https://docs.google.com/spreadsheets/d/1xQJsACKpuro7ReS3RGYlTHNxuw4o7TUoN4E6i2s_Iwo/edit#gid=0
    }
}
