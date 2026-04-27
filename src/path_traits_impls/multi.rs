//! `ClothoidPath` — a sequence of `ArcSegment`s implementing `SegmentedPath`.
//!
//! Provides `from_segments` and `from_params` constructors.

use path_traits::{ParametricPath, Path, PathError, SegmentedPath};

use crate::optimizer::Pose;
use crate::path_traits_impls::segment::ArcSegment;

/// A compound path made of linear and clothoid arc segments.
#[derive(Debug, Clone)]
pub struct ClothoidPath {
    segments: Vec<ArcSegment>,
    /// Prefix-sum of segment lengths: `prefix[i]` = sum of lengths of segments 0..i.
    prefix: Vec<f64>,
}

impl ClothoidPath {
    /// Creates a `ClothoidPath` from a vector of segments.
    ///
    /// Precomputes the prefix-sum of segment lengths for efficient `locate`.
    pub fn from_segments(segments: Vec<ArcSegment>) -> Self {
        let mut prefix = Vec::with_capacity(segments.len() + 1);
        prefix.push(0.0);
        let mut sum = 0.0;
        for seg in &segments {
            sum += seg.segment_length();
            prefix.push(sum);
        }
        Self { segments, prefix }
    }

    /// Creates a `ClothoidPath` from a flat parameter vector, mirroring
    /// `optimizer::path::eval_path_segmented`.
    ///
    /// Parameter layout: `[l₀, ks₀, ke₀, clen₀, l₁, ks₁, ke₁, clen₁, …, lₙ]`
    /// Length: `4 * n_clothoids + 1`.
    pub fn from_params(params: &[f64], n_clothoids: usize, start: Pose, n_steps: usize) -> Self {
        use crate::optimizer::ClothoidState;
        use crate::optimizer::integrate::integrate_clothoid;

        let mut segments = Vec::with_capacity(n_clothoids * 2 + 1);
        let mut current_pose = start;

        for i in 0..n_clothoids {
            let base = 4 * i;
            let l = params[base].max(0.0);
            let ks = params[base + 1];
            let ke = params[base + 2];
            let clen = params[base + 3].max(1e-6);

            if l > 1e-10 {
                segments.push(ArcSegment::Linear(
                    crate::path_traits_impls::linear::LinearSegment {
                        start: current_pose,
                        length: l,
                    },
                ));
                current_pose = Pose::new(
                    current_pose.x + l * current_pose.angle.cos(),
                    current_pose.y + l * current_pose.angle.sin(),
                    current_pose.angle,
                );
            }

            if clen > 1e-10 {
                let states: Vec<ClothoidState> = integrate_clothoid(
                    current_pose.x,
                    current_pose.y,
                    current_pose.angle,
                    ks,
                    ke,
                    clen,
                    n_steps,
                );
                if let Some(last) = states.last() {
                    let arc = crate::path_traits_impls::arc::ClothoidArc {
                        start: current_pose,
                        ks,
                        ke,
                        length: clen,
                        n_steps,
                    };
                    segments.push(ArcSegment::Clothoid(arc));
                    current_pose = Pose::new(last.x, last.y, last.theta);
                }
            }
        }

        let l_final = params[4 * n_clothoids].max(0.0);
        if l_final > 1e-10 {
            segments.push(ArcSegment::Linear(
                crate::path_traits_impls::linear::LinearSegment {
                    start: current_pose,
                    length: l_final,
                },
            ));
        }

        if segments.is_empty() {
            segments.push(ArcSegment::Linear(
                crate::path_traits_impls::linear::LinearSegment {
                    start: current_pose,
                    length: 0.0,
                },
            ));
        }

        Self::from_segments(segments)
    }
}

impl Path for ClothoidPath {
    type Point = crate::Point2;
    type Error = PathError<f64>;
    type Scalar = f64;

    fn length(&self) -> f64 {
        *self.prefix.last().unwrap_or(&0.0)
    }

    fn sample_at(&self, s: f64) -> Result<Self::Point, Self::Error> {
        let total = self.length();
        if s < 0.0 || s > total {
            return Err(PathError::out_of_domain(s, self.domain()));
        }
        let (idx, local_s) = self.locate(s)?;
        self.segments[idx].sample_at(local_s)
    }
}

impl ParametricPath for ClothoidPath {
    fn sample_t(&self, t: f64) -> Result<Self::Point, Self::Error> {
        if !(0.0..=1.0).contains(&t) {
            return Err(PathError::out_of_domain(t, 0.0..=1.0));
        }
        self.sample_at(t * self.length())
    }
}

impl SegmentedPath for ClothoidPath {
    type Segment = ArcSegment;

    fn segment_count(&self) -> usize {
        self.segments.len()
    }

    fn segments(&self) -> impl Iterator<Item = &Self::Segment> {
        self.segments.iter()
    }

    fn locate(&self, s: f64) -> Result<(usize, f64), Self::Error> {
        let total = self.length();
        if s < 0.0 || s > total {
            return Err(PathError::out_of_domain(s, self.domain()));
        }

        let n = self.segments.len();
        if n == 0 {
            return Err(PathError::out_of_domain(s, self.domain()));
        }

        let idx = if s == total {
            n - 1
        } else {
            self.prefix.partition_point(|&p| p <= s).saturating_sub(1)
        };
        let seg_start = self.prefix[idx];
        let local_s = (s - seg_start)
            .max(0.0)
            .min(self.segments[idx].segment_length());
        Ok((idx, local_s))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_path() -> ClothoidPath {
        use path_traits::Path;

        let clothoid = crate::path_traits_impls::arc::ClothoidArc {
            start: Pose::new(2.0, 0.0, 0.0),
            ks: 0.0,
            ke: 1.0,
            length: 3.0,
            n_steps: 64,
        };
        let clothoid_end_pos = clothoid.sample_at(clothoid.length).unwrap();
        let clothoid_end_angle = clothoid.heading_at_s(clothoid.length);
        let clothoid_end = Pose::new(clothoid_end_pos.x, clothoid_end_pos.y, clothoid_end_angle);

        let segs = vec![
            ArcSegment::Linear(crate::path_traits_impls::linear::LinearSegment {
                start: Pose::new(0.0, 0.0, 0.0),
                length: 2.0,
            }),
            ArcSegment::Clothoid(clothoid),
            ArcSegment::Linear(crate::path_traits_impls::linear::LinearSegment {
                start: clothoid_end,
                length: 1.0,
            }),
        ];
        ClothoidPath::from_segments(segs)
    }

    #[test]
    fn clothoid_path_segment_count() {
        let path = make_path();
        assert_eq!(path.segment_count(), 3);
    }

    #[test]
    fn clothoid_path_total_length() {
        let path = make_path();
        assert!((path.length() - 6.0).abs() < 1e-10);
    }

    #[test]
    fn clothoid_path_locate_roundtrip() {
        let path = make_path();
        for s in [0.0, 1.0, 2.0, 3.5, 5.0, 6.0] {
            let (idx, local_s) = path.locate(s).unwrap();
            let prefix_before = path.prefix[idx];
            assert!((prefix_before + local_s - s).abs() < 1e-6);
        }
    }

    #[test]
    fn clothoid_path_sample_at_matches_segment() {
        let path = make_path();
        let pt = path.sample_at(1.0).unwrap();
        let pt2 = path.segments[0].sample_at(1.0).unwrap();
        assert!((pt.x - pt2.x).abs() < 1e-10);
        assert!((pt.y - pt2.y).abs() < 1e-10);
    }

    #[test]
    fn clothoid_path_from_params_single_clothoid() {
        let start = Pose::new(0.0, 0.0, 0.0);
        let params = [0.0, 0.0, 0.0, 5.0, 0.0];
        let path = ClothoidPath::from_params(&params, 1, start, 100);
        assert!((path.length() - 5.0).abs() < 1e-10);
        let end = path.end().unwrap();
        assert!((end.x - 5.0).abs() < 1e-6);
        assert!(end.y.abs() < 1e-6);
    }

    #[test]
    fn clothoid_path_from_params_with_prefix_straight() {
        let start = Pose::new(0.0, 0.0, 0.0);
        let params = [2.0, 0.0, 0.0, 3.0, 0.0];
        let path = ClothoidPath::from_params(&params, 1, start, 100);
        assert_eq!(path.segment_count(), 2);
        assert!((path.length() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn clothoid_path_from_params_matches_eval_path_endpoint() {
        let start = Pose::new(0.0, 0.0, 0.0);
        let params = [0.0, 0.0, 0.0, 5.0, 0.0];
        let path = ClothoidPath::from_params(&params, 1, start, 100);
        let path_end = path.end().unwrap();

        let eval_pts = crate::optimizer::path::eval_path(&params, 1, &start, 100);
        let eval_end = eval_pts.last().unwrap();

        assert!((path_end.x - eval_end.x).abs() < 1e-6);
        assert!((path_end.y - eval_end.y).abs() < 1e-6);
    }

    #[test]
    fn clothoid_path_parametric() {
        let path = make_path();
        let pt = path.sample_t(0.5).unwrap();
        let pt2 = path.sample_at(3.0).unwrap();
        assert!((pt.x - pt2.x).abs() < 1e-6);
        assert!((pt.y - pt2.y).abs() < 1e-6);
    }

    #[test]
    fn clothoid_path_out_of_domain() {
        let path = make_path();
        assert!(path.sample_at(-0.1).is_err());
        assert!(path.sample_at(6.1).is_err());
    }

    #[test]
    fn clothoid_path_start() {
        let path = make_path();
        let s = path.start().unwrap();
        assert!((s.x - 0.0).abs() < 1e-10);
        assert!((s.y - 0.0).abs() < 1e-10);
    }

    #[test]
    fn clothoid_path_reverse_roundtrip() {
        use path_traits::PathExt;
        let path = make_path();
        let original_samples: Vec<_> = (0..=20)
            .map(|i| path.sample_at(i as f64 * path.length() / 20.0).unwrap())
            .collect();
        let reversed = path.reverse();
        let reversed_again = reversed.reverse();
        let rev_samples: Vec<_> = (0..=20)
            .map(|i| {
                reversed_again
                    .sample_at(i as f64 * reversed_again.length() / 20.0)
                    .unwrap()
            })
            .collect();
        for (a, b) in original_samples.iter().zip(rev_samples.iter()) {
            assert!((a.x - b.x).abs() < 1e-6);
            assert!((a.y - b.y).abs() < 1e-6);
        }
    }

    #[test]
    fn clothoid_path_concat_length() {
        use path_traits::PathExt;
        let path = make_path();
        let other = make_path();
        let concat = path.concat(other);
        let expected = concat.length();
        assert!((expected - 12.0).abs() < 1e-6);
    }
}
