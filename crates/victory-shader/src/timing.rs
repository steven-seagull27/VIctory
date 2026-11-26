/// Completed calls, collected only when the translation's debug event is enabled.
#[derive(Default)]
pub(crate) struct Timings {
    /// Phase name, accumulated microseconds and number of calls.
    #[cfg(feature = "diagnostics")]
    pub(crate) samples: Option<Vec<(&'static str, u128, usize)>>,
}

impl Timings {
    pub(crate) fn for_translation() -> Self {
        Self {
            #[cfg(feature = "diagnostics")]
            samples: tracing::enabled!(target: "victory_shader::emit", tracing::Level::DEBUG)
                .then(Vec::new),
        }
    }

    #[inline]
    pub(crate) fn measure<T>(&mut self, phase: &'static str, run: impl FnOnce() -> T) -> T {
        #[cfg(feature = "diagnostics")]
        if let Some(samples) = &mut self.samples {
            let start = std::time::Instant::now();
            let result = run();
            let us = start.elapsed().as_micros();
            if let Some((_, total, calls)) = samples.iter_mut().find(|s| s.0 == phase) {
                *total += us;
                *calls += 1;
            } else {
                samples.push((phase, us, 1));
            }
            return result;
        }
        let _ = phase;
        run()
    }
}
