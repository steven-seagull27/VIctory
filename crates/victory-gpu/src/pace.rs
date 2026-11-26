//! Present pacing (`VICTORY_FPS=<frames per second>`): the title's present
//! runs on its own thread and the host shows each finished frame at once
//! (`Present(0)`, no sync interval), so on a strong GPU frames are made as
//! fast as the title's loop allows — the card heats, a laptop throttles
//! mid-play, and the frame times jitter the play. With the variable set,
//! the title's present waits for the next frame's deadline instead, so a
//! frame lands about every period: `VICTORY_FPS=30` is the console's own
//! cadence. Without it, nothing waits that did not before.

#![cfg(windows)]

use std::time::{Duration, Instant};

use windows_sys::Win32::Foundation::WAIT_FAILED;
use windows_sys::Win32::System::Threading::{
    CreateWaitableTimerExW, SetWaitableTimer, WaitForSingleObject,
    CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, INFINITE, TIMER_ALL_ACCESS,
};

/// The last milliseconds a wake leaves for a spin instead of a sleep:
/// a sleep's wake is late by up to a timer tick, a spin is not.
const SPIN: Duration = Duration::from_millis(2);

/// The paced frame's wait from a `VICTORY_FPS` value: a cadence of at
/// least one and at most a thousand frames per second.
fn period_of(value: Option<&str>) -> Option<Duration> {
    let hz = value?.trim().parse::<u32>().ok()?;
    if !(1..=1000).contains(&hz) {
        return None;
    }
    Some(Duration::from_nanos(1_000_000_000 / u64::from(hz)))
}

/// A high-resolution waitable timer, or none where the OS does not make
/// one (a plain sleep's wake is within a tick, and a spin makes up for it).
fn high_resolution_timer() -> Option<std::os::windows::io::OwnedHandle> {
    use std::os::windows::io::{FromRawHandle, OwnedHandle};
    // SAFETY: a plain creation of a nameless timer object.
    let handle = unsafe {
        CreateWaitableTimerExW(
            std::ptr::null(),
            std::ptr::null(),
            CREATE_WAITABLE_TIMER_HIGH_RESOLUTION,
            TIMER_ALL_ACCESS,
        )
    };
    if handle.is_null() {
        return None;
    }
    // SAFETY: the handle is ours, created just above.
    Some(unsafe { OwnedHandle::from_raw_handle(handle as *mut _) })
}

/// Sleeps to the `deadline` — the timer when there is one, else a plain
/// sleep — and spins the last [`SPIN`], so the wake is on time.
fn sleep_until(deadline: Instant, timer: Option<&std::os::windows::io::OwnedHandle>) {
    use std::os::windows::io::AsRawHandle;
    let left = deadline.saturating_duration_since(Instant::now());
    if left > SPIN {
        let due = -(((left - SPIN).as_nanos() as i64 + 99) / 100); // 100 ns units, relative
        match timer {
            // SAFETY: our timer handle, a plain relative due time.
            Some(handle) => unsafe {
                let on = SetWaitableTimer(
                    handle.as_raw_handle(),
                    &due,
                    0,
                    None,
                    std::ptr::null(),
                    0,
                ) != 0;
                // The fallback sleeps where the timer left nothing waiting:
                // a failed arm of `SetWaitableTimer` (or of the wait itself)
                // would busy-spin through the whole frame's rest.
                if !on || WaitForSingleObject(handle.as_raw_handle(), INFINITE) == WAIT_FAILED {
                    std::thread::sleep(left - SPIN);
                }
            },
            None => std::thread::sleep(left - SPIN),
        }
    }
    while Instant::now() < deadline {
        std::thread::yield_now();
    }
}

/// The presents' pacer, from `VICTORY_FPS`.
pub(crate) struct Pacer {
    period: Option<Duration>,
    timer: Option<std::os::windows::io::OwnedHandle>,
    /// The next frame's deadline once a present happened.
    next: Option<Instant>,
}

impl Pacer {
    /// The pacer the `VICTORY_FPS` variable asks for (none when unset,
    /// none on a value that is no cadence, logged then).
    pub(crate) fn from_env() -> Pacer {
        let value = std::env::var("VICTORY_FPS").ok();
        match period_of(value.as_deref()) {
            Some(period) => {
                tracing::info!(fps = value, ms = period.as_millis() as u64, "presents are paced");
                Pacer {
                    period: Some(period),
                    timer: high_resolution_timer(),
                    next: None,
                }
            }
            None if value.is_some() => {
                tracing::warn!(fps = value, "VICTORY_FPS is no frame count; presents are not paced");
                Pacer { period: None, timer: None, next: None }
            }
            None => Pacer { period: None, timer: None, next: None },
        }
    }

    /// Waits for the next paced frame's deadline: frames kept on time
    /// follow their chain (a present per period; the first present starts
    /// the chain, it does not wait), and one late present restarts the
    /// chain from now, so nothing catches up by bursting.
    pub(crate) fn wait(&mut self) {
        self.wait_with(Instant::now, sleep_until);
    }

    fn wait_with(
        &mut self,
        now: impl FnOnce() -> Instant,
        wait_until: impl FnOnce(Instant, Option<&std::os::windows::io::OwnedHandle>),
    ) {
        let Some(period) = self.period else { return };
        if let Some(deadline) = self.next {
            wait_until(deadline, self.timer.as_ref());
        }
        // A timer wake or the yielding thread can miss the deadline.
        // Start the next interval when this wait actually completes.
        self.next = Some(now() + period);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_late_wake_leaves_a_full_period_before_the_next_frame() {
        use std::cell::Cell;
        let period = Duration::from_millis(20);
        let start = Instant::now();
        let clock = Cell::new(start);
        let mut pacer = Pacer { period: Some(period), timer: None, next: None };
        let mut frame = |delay_ms| {
            pacer.wait_with(
                || clock.get(),
                |deadline, _| {
                    clock.set(clock.get().max(deadline) + Duration::from_millis(delay_ms));
                },
            );
            clock.get() - start
        };
        for (wake_delay, expected_ms) in [(0, 0), (0, 20), (9, 49), (0, 69), (40, 129), (0, 149)] {
            assert_eq!(frame(wake_delay), Duration::from_millis(expected_ms));
        }
    }

    #[test]
    fn the_fps_value_must_be_a_cadence() {
        assert_eq!(period_of(None), None);
        assert_eq!(period_of(Some("")), None);
        assert_eq!(period_of(Some("abc")), None);
        assert_eq!(period_of(Some("0")), None);
        assert_eq!(period_of(Some("1001")), None);
        assert_eq!(period_of(Some("30")), Some(Duration::from_nanos(33_333_333)));
        assert_eq!(period_of(Some("60")), Some(Duration::from_nanos(16_666_666)));
        // Padded and improper case values hold.
        assert_eq!(period_of(Some(" 30 ")), Some(Duration::from_nanos(33_333_333)));
    }

    /// Two presents a pacer of a cadence keeps apart by about the period.
    #[test]
    fn a_paced_present_waits_for_its_deadline() {
        let period = Duration::from_millis(40);
        let mut pacer =
            Pacer { period: Some(period), timer: None, next: None };
        let started = Instant::now();
        pacer.wait();
        let first = started.elapsed();
        // The first present does not wait (the chain starts from it).
        assert!(first < period, "{first:?}");
        pacer.wait();
        let second = started.elapsed();
        assert!(second >= period, "{second:?}");
        assert!(second < period * 4, "{second:?}");
    }

    /// A late present does not burst to catch up: the chain restarts.
    #[test]
    fn a_late_present_restarts_its_chain() {
        let period = Duration::from_millis(20);
        let mut pacer =
            Pacer { period: Some(period), timer: None, next: None };
        pacer.wait();
        // Time passes without a present: the next wait is not a burst of
        // several missed deadlines, only until the next one.
        std::thread::sleep(period * 3);
        let started = Instant::now();
        pacer.wait();
        assert!(started.elapsed() < period * 2, "{:?}", started.elapsed());
    }
}
