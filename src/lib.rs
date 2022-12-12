use fresnel::fresnl;

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
    const PI_SQRT: f64 = 1.7724538509055160272981674833411451827975494561223871282138077898f64;
    const INV_PI_SQRT: f64 = 0.5641895835477562869480794515607725858440506293289988568440857217f64;

    pub fn new(a: f64) -> Self {
        Self { a }
    }

    #[inline]
    pub fn direction_angle(&self, arc_length: f64) -> f64 {
        0.5 * (arc_length * arc_length) / (self.a * self.a)
    }

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
        let (s, c) = fresnl(t * Self::INV_PI_SQRT);
        Point2 {
            x: self.a * Self::PI_SQRT * s,
            y: self.a * Self::PI_SQRT * c,
        }
    }

    fn calculate_approx(&self, t: f64) -> Point2 {
        let fsc = FresnelSinCos::compute(t * Self::INV_PI_SQRT);
        Point2 {
            x: self.a * Self::PI_SQRT * fsc.sin,
            y: self.a * Self::PI_SQRT * fsc.cos,
        }
    }
}

struct AuxFg {
    pub f: f64,
    pub g: f64,
}

impl AuxFg {
    pub fn compute(x: f64) -> Self {
        // debug_assert_ne!(x, 0.);

        let x2 = x * x;
        let x3 = x * x * x;

        let f = (1. + 0.926 * x) / (2. + 1.792 * x + 3.104 * x2);
        let g = 1. / (2. + 4.142 * x + 3.492 * x2 + 6.670 * x3);

        Self { f, g }
    }
}

struct FresnelSinCos {
    pub cos: f64,
    pub sin: f64,
}

impl FresnelSinCos {
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
    #[cfg(feature = "fresnel")]
    fn calculate_fresnl_works() {
        let clothoid = Clothoid::new(8.);
        let pt = clothoid.calculate_fresnl(0.);
        assert_f64_near!(pt.x, 0.);
        assert_f64_near!(pt.y, 0.);

        let pt = clothoid.calculate_fresnl(std::f64::consts::PI);
        assert_f64_near!(pt.x, 6.7669799976205);
        assert_f64_near!(pt.y, 4.615663254508842);
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
}
