//! What becomes of an audio stream's errors, so a failing one cannot spin (#507).
//!
//! cpal 0.17's ALSA backend reported an underrun that left the device in its
//! XRUN state as "`alsa::poll()` returned POLLERR" and polled again without
//! recovering it: the error came straight back, tens of thousands of times a
//! second, for as long as the stream lived, and rodio's default error callback
//! printed each one to stderr. cpal 0.18 recovers that one, but a backend that
//! repeats an error must still not hold a core or fill the log, so every
//! stream ZapFast opens reports to a [`Health`] instead. It logs the first error and then a count at most every
//! [`LOG_EVERY`], and once errors arrive faster than a working stream reports
//! them it marks the stream failed and slows the audio thread down, until the
//! stream's owner sees [`Health::failed`], drops it and opens another.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use rodio::cpal::{Error, ErrorKind};

/// Errors within [`FLOOD_WINDOW`] that make a stream failed. A working
/// stream reports an underrun now and then, never dozens in a second.
const FLOOD: u32 = 50;
const FLOOD_WINDOW: Duration = Duration::from_secs(1);
/// How often a stream that keeps reporting errors is logged again.
const LOG_EVERY: Duration = Duration::from_secs(10);
/// How long the audio thread waits after each error once its stream failed,
/// so a backend that keeps reporting one cannot hold a core until the owner
/// drops the stream. Dropping waits for the thread, so this stays short.
const FAILED_PAUSE: Duration = Duration::from_millis(20);

/// Watches one audio stream's errors. Cloned into its error callback.
#[derive(Clone)]
pub struct Health {
    inner: Arc<Inner>,
}

struct Inner {
    /// Names the stream in the log, such as "voice playback".
    label: &'static str,
    failed: AtomicBool,
    tally: Mutex<Tally>,
    /// Tells the owner, which may otherwise not look until its next frame.
    wake: Option<Box<dyn Fn() + Send + Sync>>,
}

impl Health {
    pub fn new(label: &'static str) -> Self {
        Self::build(label, None)
    }

    /// Calls `wake` once, when the stream fails.
    pub fn waking(label: &'static str, wake: impl Fn() + Send + Sync + 'static) -> Self {
        Self::build(label, Some(Box::new(wake)))
    }

    fn build(label: &'static str, wake: Option<Box<dyn Fn() + Send + Sync>>) -> Self {
        Self {
            inner: Arc::new(Inner {
                label,
                failed: AtomicBool::new(false),
                tally: Mutex::new(Tally::default()),
                wake,
            }),
        }
    }

    /// Whether the stream keeps failing and should be dropped and reopened.
    pub fn failed(&self) -> bool {
        self.inner.failed.load(Ordering::Relaxed)
    }

    /// The error callback to hand rodio for this stream.
    pub fn callback(&self) -> impl FnMut(Error) + Send + Clone + 'static {
        let health = self.clone();
        move |error| health.report(&error, Instant::now())
    }

    fn report(&self, error: &Error, now: Instant) {
        let inner = &self.inner;
        let verdict = inner
            .tally
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .record(error, now);
        if let Some(line) = verdict.log {
            log::warn!("{}: {line}", inner.label);
        }
        if verdict.fail && !inner.failed.swap(true, Ordering::Relaxed) {
            log::warn!(
                "{}: the audio stream keeps failing; closing it",
                inner.label
            );
            if let Some(wake) = &inner.wake {
                wake();
            }
        }
        if self.failed() {
            std::thread::sleep(FAILED_PAUSE);
        }
    }
}

/// What to do about one error.
#[derive(Debug, Default, PartialEq)]
struct Verdict {
    log: Option<String>,
    fail: bool,
}

/// The errors one stream has reported.
#[derive(Default)]
struct Tally {
    window_start: Option<Instant>,
    in_window: u32,
    logged_at: Option<Instant>,
    /// Errors since the last line logged.
    unlogged: u64,
}

impl Tally {
    fn record(&mut self, error: &Error, now: Instant) -> Verdict {
        // Notices about a stream that plays on are logged, not counted.
        let notice = matches!(
            error.kind(),
            ErrorKind::DeviceChanged | ErrorKind::RealtimeDenied
        );
        match self.window_start {
            _ if notice => {}
            Some(start) if now.saturating_duration_since(start) < FLOOD_WINDOW => {
                self.in_window += 1;
            }
            _ => {
                self.window_start = Some(now);
                self.in_window = 1;
            }
        }
        // A stream cpal says is over is not one to wait on.
        let over = matches!(
            error.kind(),
            ErrorKind::DeviceNotAvailable
                | ErrorKind::StreamInvalidated
                | ErrorKind::HostUnavailable
        );
        let fail = over || self.in_window >= FLOOD;
        let log = match self.logged_at {
            None => Some(format!("audio stream error: {error}")),
            Some(at) if now.saturating_duration_since(at) >= LOG_EVERY => Some(format!(
                "audio stream error: {error} ({} more since the last report)",
                self.unlogged
            )),
            _ => None,
        };
        if log.is_some() {
            self.logged_at = Some(now);
            self.unlogged = 0;
        } else {
            self.unlogged += 1;
        }
        Verdict { log, fail }
    }
}

/// The error cpal 0.17 repeated in #507, as cpal 0.18 would word it.
#[cfg(test)]
pub(crate) fn pollerr() -> Error {
    Error::with_message(ErrorKind::BackendError, "`alsa::poll()` returned POLLERR")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_occasional_underrun_is_logged_and_keeps_the_stream() {
        let mut tally = Tally::default();
        let start = Instant::now();
        let first = tally.record(&Error::new(ErrorKind::Xrun), start);
        assert!(first.log.is_some());
        assert!(!first.fail);
        for second in 1..30 {
            let verdict = tally.record(
                &Error::new(ErrorKind::Xrun),
                start + Duration::from_secs(second),
            );
            assert!(!verdict.fail, "one underrun a second is not a flood");
        }
    }

    /// The flood of #507: cpal 0.17 reporting POLLERR in a loop, about 28,000
    /// times a second, is logged a handful of times and fails the stream
    /// within its first fifty errors.
    #[test]
    fn a_pollerr_flood_fails_the_stream_and_is_logged_sparingly() {
        let mut tally = Tally::default();
        let start = Instant::now();
        let step = Duration::from_secs(1) / 28_000;
        let mut logged = 0;
        let mut failed_at = None;
        for index in 0..28_000 * 30 {
            let verdict = tally.record(&pollerr(), start + step * index);
            logged += usize::from(verdict.log.is_some());
            if verdict.fail && failed_at.is_none() {
                failed_at = Some(index);
            }
        }
        assert_eq!(failed_at, Some(FLOOD - 1));
        assert!(logged <= 4, "logged {logged} lines in 30 seconds");
    }

    #[test]
    fn a_later_report_counts_what_was_not_logged() {
        let mut tally = Tally::default();
        let start = Instant::now();
        tally.record(&pollerr(), start);
        for index in 1..=5 {
            assert_eq!(
                tally.record(&pollerr(), start + Duration::from_millis(index)),
                Verdict::default()
            );
        }
        let line = tally
            .record(&pollerr(), start + LOG_EVERY)
            .log
            .expect("logged again after the interval");
        assert!(line.contains("5 more"), "{line}");
    }

    #[test]
    fn a_rerouted_stream_is_not_counted_towards_a_flood() {
        let mut tally = Tally::default();
        let now = Instant::now();
        for _ in 0..FLOOD * 2 {
            assert!(
                !tally
                    .record(&Error::new(ErrorKind::DeviceChanged), now)
                    .fail
            );
        }
    }

    #[test]
    fn a_device_that_went_away_fails_at_once() {
        let mut tally = Tally::default();
        assert!(
            tally
                .record(&Error::new(ErrorKind::DeviceNotAvailable), Instant::now())
                .fail
        );
    }

    #[test]
    fn a_failed_stream_wakes_its_owner_once_and_slows_its_thread() {
        let woken = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let health = {
            let woken = Arc::clone(&woken);
            Health::waking("test", move || {
                woken.fetch_add(1, Ordering::Relaxed);
            })
        };
        let mut callback = health.callback();
        for _ in 0..FLOOD - 1 {
            callback(pollerr());
        }
        assert!(!health.failed());
        let started = Instant::now();
        callback(pollerr());
        callback(pollerr());
        assert!(health.failed());
        assert_eq!(woken.load(Ordering::Relaxed), 1);
        assert!(started.elapsed() >= FAILED_PAUSE * 2);
    }
}
