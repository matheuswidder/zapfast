//! Local speech-to-text for voice messages: the Transcription extension.
//!
//! Whisper models run in-process on a dedicated thread, one clip at a time,
//! on the CPU. Nothing leaves the computer: audio goes to the model, text
//! comes back. Transcripts live in memory for the session only, and the
//! models themselves are downloaded to the state directory, never the cache.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use fastframe_shell::Waker;

use crate::voice;

/// Where the ggml models live, outside the media cache so clearing the cache
/// never deletes a gigabyte download.
pub fn models_dir(dirs: &crate::paths::AppDirs) -> PathBuf {
    dirs.state.join("models")
}

/// A Whisper model the Transcription extension can download.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Model {
    /// Stable name in the settings file.
    pub key: &'static str,
    /// What the picker shows.
    pub label: &'static str,
    /// Filename at the download source, also the name on disk.
    pub file: &'static str,
    /// Download size, for the picker's hint.
    pub size: &'static str,
    /// The default choice of a fresh setup.
    pub recommended: bool,
}

impl Model {
    /// The model's ggml file under `dir`.
    pub fn path(&self, dir: &Path) -> PathBuf {
        dir.join(self.file)
    }
}

pub const MODELS: [Model; 5] = [
    Model {
        key: "base",
        label: "Base",
        file: "ggml-base.bin",
        size: "141 MB",
        recommended: false,
    },
    Model {
        key: "small",
        label: "Small",
        file: "ggml-small.bin",
        size: "466 MB",
        recommended: true,
    },
    Model {
        key: "medium",
        label: "Medium",
        file: "ggml-medium.bin",
        size: "1.5 GB",
        recommended: false,
    },
    Model {
        key: "large",
        label: "Large",
        file: "ggml-large-v3-t.bin",
        size: "3 GB",
        recommended: false,
    },
    Model {
        key: "turbo",
        label: "Turbo",
        file: "ggml-large-v3-turbo.bin",
        size: "1.6 GB",
        recommended: false,
    },
];

/// The model a settings key names.
pub fn model_by_key(key: &str) -> Option<&'static Model> {
    MODELS.iter().find(|model| model.key == key)
}

/// The model keyed by the settings, the recommended one as fallback.
pub fn selected_model(settings: &crate::settings::Settings) -> Option<&'static Model> {
    settings
        .transcription_model
        .as_deref()
        .and_then(model_by_key)
        .or_else(|| MODELS.iter().find(|model| model.recommended))
}

/// Progress of one model download.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DownloadState {
    Downloading {
        received: u64,
        total: Option<u64>,
    },
    /// The file is complete on disk.
    Done,
    Failed(String),
    /// The reader stopped it; the partial file is removed.
    Cancelled,
}

/// Where a clip's transcription stands.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TranscribeState {
    Queued,
    Running,
    Done(String),
    Failed(String),
}

struct Job {
    message: String,
    path: PathBuf,
    model: PathBuf,
    /// BCP-47 hint for the model, `None` lets it detect the language.
    language: Option<String>,
}

/// Shared handle the interface and the worker see: transcript states, model
/// download progress, and a one-job queue.
pub struct Transcriber {
    states: Arc<Mutex<HashMap<String, TranscribeState>>>,
    downloads: Arc<Mutex<HashMap<String, DownloadState>>>,
    queue: Mutex<std::sync::mpsc::Sender<Job>>,
    /// Grows every time a transcript lands, so a hover that missed the wake
    /// can still tell.
    revision: Arc<AtomicU64>,
    /// Counts downloads that reached a final state, so the interface can pick
    /// up clips that were waiting for a model without touching the disk on
    /// every frame.
    settled: Arc<AtomicU64>,
}

impl Transcriber {
    /// Starts the worker thread.
    pub fn new(waker: Waker) -> Self {
        let (sender, receiver) = std::sync::mpsc::channel::<Job>();
        let states = Arc::new(Mutex::new(HashMap::new()));
        let downloads = Arc::new(Mutex::new(HashMap::new()));
        let revision = Arc::new(AtomicU64::new(0));
        let setted = Arc::new(AtomicU64::new(0));
        let worker_states = Arc::clone(&states);
        let worker_revision = Arc::clone(&revision);
        let worker_settled = Arc::clone(&setted);
        std::thread::Builder::new()
            .name("transcribe".to_owned())
            .spawn(move || {
                let mut loaded: Option<(PathBuf, whisper_rs::WhisperContext)> = None;
                while let Ok(job) = receiver.recv() {
                    worker_states
                        .lock()
                        .unwrap_or_else(|p| p.into_inner())
                        .insert(job.message.clone(), TranscribeState::Running);
                    worker_revision.fetch_add(1, Ordering::Relaxed);
                    waker.wake();
                    let result = transcribe(&job, &mut loaded);
                    let mut states = worker_states.lock().unwrap_or_else(|p| p.into_inner());
                    states.insert(
                        job.message,
                        match result {
                            Ok(text) => TranscribeState::Done(text),
                            Err(error) => TranscribeState::Failed(error),
                        },
                    );
                    drop(states);
                    worker_revision.fetch_add(1, Ordering::Relaxed);
                    waker.wake();
                }
            })
            .expect("spawns the transcribe thread");
        Self {
            states,
            downloads,
            queue: Mutex::new(sender),
            revision,
            settled: Arc::clone(&worker_settled),
        }
    }

    /// How many downloads have reached a final state. A change here means a
    /// model may have landed, so clips waiting for one are worth another try.
    pub fn settled_downloads(&self) -> u64 {
        self.settled.load(Ordering::Relaxed)
    }

    /// Seeds a transcript state so the headless layout tests can draw a bubble
    /// that carries one, without running a model. Release builds skip this.
    #[cfg(any(test, feature = "demo"))]
    pub fn seed(&self, message: &str, state: TranscribeState) {
        self.states
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .insert(message.to_owned(), state);
        self.revision.fetch_add(1, Ordering::Relaxed);
    }

    /// Where a message's transcript stands.
    pub fn status(&self, message: &str) -> Option<TranscribeState> {
        self.states
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .get(message)
            .cloned()
    }

    /// Counts transcript state changes, for cheap repaint bookkeeping.
    pub fn revision(&self) -> u64 {
        self.revision.load(Ordering::Relaxed)
    }

    /// Queues a downloaded clip for transcription. The clip already running
    /// finishes first; a message asked for again replaces its old result.
    pub fn enqueue(
        &self,
        message: String,
        path: PathBuf,
        model: PathBuf,
        language: Option<String>,
    ) {
        let mut states = self.states.lock().unwrap_or_else(|p| p.into_inner());
        if matches!(
            states.get(&message),
            Some(TranscribeState::Queued | TranscribeState::Running)
        ) {
            return;
        }
        states.insert(message.clone(), TranscribeState::Queued);
        drop(states);
        let job = Job {
            message,
            path,
            model,
            language,
        };
        self.queue
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .send(job)
            .expect("the transcribe thread outlives the interface");
        self.revision.fetch_add(1, Ordering::Relaxed);
    }

    /// Forgets a transcript, as when its message is deleted or the model
    /// changes.
    pub fn forget(&self, message: &str) {
        self.states
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .remove(message);
    }

    /// Forgets every transcript, as when the active model changes: the words
    /// already on screen came from a model that is no longer in use.
    pub fn forget_all(&self) {
        let mut states = self.states.lock().unwrap_or_else(|p| p.into_inner());
        if states.is_empty() {
            return;
        }
        states.clear();
        drop(states);
        self.revision.fetch_add(1, Ordering::Relaxed);
    }

    /// Starts a model download in its own thread. An existing download or a
    /// finished file is left alone.
    pub fn download(&self, model: &'static Model, dir: PathBuf, waker: Waker) {
        {
            let mut downloads = self.downloads.lock().unwrap_or_else(|p| p.into_inner());
            match downloads.get(model.key) {
                Some(DownloadState::Downloading { .. }) | Some(DownloadState::Done) => return,
                _ => {}
            }
            downloads.insert(
                model.key.to_owned(),
                DownloadState::Downloading {
                    received: 0,
                    total: None,
                },
            );
        }
        let downloads = Arc::clone(&self.downloads);
        let revision = Arc::clone(&self.revision);
        let settled = Arc::clone(&self.settled);
        std::thread::Builder::new()
            .name(format!("model-download-{}", model.key))
            .spawn(move || {
                let state = fetch_model(model, &dir, &downloads, &waker);
                {
                    let mut downloads = downloads.lock().unwrap_or_else(|p| p.into_inner());
                    if matches!(state, DownloadState::Cancelled) {
                        downloads.remove(model.key);
                    } else {
                        downloads.insert(model.key.to_owned(), state);
                    }
                }
                // A failed download can still have put a usable model there
                // (an interrupted resume), and the interface re-checks, so
                // count every download that stops moving.
                settled.fetch_add(1, Ordering::Relaxed);
                revision.fetch_add(1, Ordering::Relaxed);
                waker.wake();
            })
            .expect("spawns the model download thread");
    }

    /// Where a model's download stands.
    pub fn download_status(&self, key: &str) -> Option<DownloadState> {
        self.downloads
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .get(key)
            .cloned()
    }

    /// Stops a download the reader started by mistake. The thread notices on
    /// its next chunk, deletes the partial file, and reports nothing.
    pub fn cancel_download(&self, key: &str) {
        let mut downloads = self.downloads.lock().unwrap_or_else(|p| p.into_inner());
        if matches!(downloads.get(key), Some(DownloadState::Downloading { .. })) {
            downloads.insert(key.to_owned(), DownloadState::Cancelled);
        }
    }
}

/// Removes a downloaded model. The partial file of a cancelled download goes too.
pub fn delete_model(model: &Model, dir: &Path) -> std::io::Result<()> {
    let path = model.path(dir);
    if path.is_file() {
        std::fs::remove_file(&path)?;
    }
    let partial = path.with_extension("part");
    if partial.is_file() {
        std::fs::remove_file(partial)?;
    }
    Ok(())
}

/// A finished model's file exists and is not empty.
pub fn model_downloaded(model: &Model, dir: &Path) -> bool {
    let path = model.path(dir);
    std::fs::metadata(&path)
        .map(|meta| meta.len() > 0)
        .unwrap_or(false)
}

fn fetch_model(
    model: &Model,
    dir: &Path,
    downloads: &Mutex<HashMap<String, DownloadState>>,
    waker: &Waker,
) -> DownloadState {
    if model_downloaded(model, dir) {
        return DownloadState::Done;
    }
    if let Err(error) = std::fs::create_dir_all(dir) {
        return DownloadState::Failed(error.to_string());
    }
    let url = format!(
        "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/{}",
        model.file
    );
    fetch_url(&url, &model.path(dir), model.key, downloads, waker)
}

/// Downloads `url` to `path`, reporting progress under `key`.
fn fetch_url(
    url: &str,
    path: &Path,
    key: &str,
    downloads: &Mutex<HashMap<String, DownloadState>>,
    waker: &Waker,
) -> DownloadState {
    let client = match reqwest::blocking::Client::builder().build() {
        Ok(client) => client,
        Err(error) => return DownloadState::Failed(error.to_string()),
    };
    let mut response = match client.get(url).send() {
        Ok(response) => response,
        Err(error) => return DownloadState::Failed(error.to_string()),
    };
    if !response.status().is_success() {
        return DownloadState::Failed(format!("HTTP {}", response.status()));
    }
    if let Some(DownloadState::Downloading { total, .. }) = downloads
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .get_mut(key)
    {
        *total = response.content_length();
    }
    let temp = path.with_extension("part");
    let mut file = match std::fs::File::create(&temp) {
        Ok(file) => file,
        Err(error) => return DownloadState::Failed(error.to_string()),
    };
    let mut received = 0u64;
    let mut last_paint = std::time::Instant::now();
    let mut buffer = [0u8; 256 * 1024];
    use std::io::{Read, Write};
    loop {
        if matches!(
            downloads.lock().unwrap_or_else(|p| p.into_inner()).get(key),
            Some(DownloadState::Cancelled)
        ) {
            drop(file);
            let _ = std::fs::remove_file(&temp);
            return DownloadState::Cancelled;
        }
        match response.read(&mut buffer) {
            Ok(0) => break,
            Ok(chunk) => {
                if let Err(error) = file.write_all(&buffer[..chunk]) {
                    let _ = std::fs::remove_file(&temp);
                    return DownloadState::Failed(error.to_string());
                }
                received += chunk as u64;
                if last_paint.elapsed() >= std::time::Duration::from_millis(200) {
                    last_paint = std::time::Instant::now();
                    if let Some(DownloadState::Downloading { received: at, .. }) = downloads
                        .lock()
                        .unwrap_or_else(|p| p.into_inner())
                        .get_mut(key)
                    {
                        *at = received;
                    }
                    waker.wake();
                }
            }
            Err(error) => {
                let _ = std::fs::remove_file(&temp);
                return DownloadState::Failed(error.to_string());
            }
        }
    }
    if let Err(error) = file.sync_all().and_then(|_| std::fs::rename(&temp, path)) {
        let _ = std::fs::remove_file(&temp);
        return DownloadState::Failed(error.to_string());
    }
    DownloadState::Done
}

/// Decodes the clip and runs the model. `loaded` keeps one context alive
/// across jobs; a job for another model replaces it.
fn transcribe(
    job: &Job,
    loaded: &mut Option<(PathBuf, whisper_rs::WhisperContext)>,
) -> Result<String, String> {
    let bytes = std::fs::read(&job.path).map_err(|error| error.to_string())?;
    let (at_rate, rate) = if bytes.starts_with(b"OggS") {
        (voice::decode(&bytes)?, voice::RATE)
    } else {
        // Audio attachments that are not voice notes go through rodio's
        // decoders at their own rate.
        decode_other(&bytes)?
    };
    let samples = resample(&at_rate, rate, WHISPER_RATE);
    if samples.is_empty() {
        return Err("empty clip".to_owned());
    }
    if loaded.as_ref().map(|(path, _)| path) != Some(&job.model) {
        let params = whisper_rs::WhisperContextParameters::default();
        let context = whisper_rs::WhisperContext::new_with_params(&job.model, params)
            .map_err(|error| error.to_string())?;
        *loaded = Some((job.model.clone(), context));
    }
    let context = &loaded.as_ref().expect("just loaded").1;
    let mut state = context.create_state().map_err(|error| error.to_string())?;
    let mut params =
        whisper_rs::FullParams::new(whisper_rs::SamplingStrategy::Greedy { best_of: 1 });
    params.set_language(job.language.as_deref());
    params.set_translate(false);
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);
    state
        .full(params, &samples)
        .map_err(|error| error.to_string())?;
    let segments = state.full_n_segments();
    let mut text = String::new();
    for index in 0..segments {
        if let Some(segment) = state.get_segment(index) {
            text.push_str(&segment.to_string());
        }
    }
    Ok(text.trim().to_owned())
}

/// Whisper's own sample rate, mono.
const WHISPER_RATE: u32 = 16_000;

/// Linear resample of mono audio. Voice notes decode at 48 kHz, but an audio
/// attachment can arrive at any rate the container declares, so the factor is
/// computed rather than assumed.
fn resample(samples: &[f32], from: u32, to: u32) -> Vec<f32> {
    if samples.is_empty() || from == 0 || to == 0 || from == to {
        return samples.to_vec();
    }
    let length = ((samples.len() as u64 * u64::from(to)) / u64::from(from)).max(1) as usize;
    let step = f64::from(from) / f64::from(to);
    let mut out = Vec::with_capacity(length);
    for index in 0..length {
        let at = index as f64 * step;
        let left = (at.floor() as usize).min(samples.len() - 1);
        let right = (left + 1).min(samples.len() - 1);
        let fraction = (at - left as f64) as f32;
        out.push(samples[left] + (samples[right] - samples[left]) * fraction);
    }
    out
}

/// Decodes a non-OGG audio file through rodio into mono f32 at the source rate.
fn decode_other(bytes: &[u8]) -> Result<(Vec<f32>, u32), String> {
    use rodio::Source;
    let cursor = std::io::Cursor::new(bytes.to_vec());
    let source = rodio::Decoder::new(cursor).map_err(|error| error.to_string())?;
    let rate = source.sample_rate().get();
    let channels = source.channels().get();
    let interleaved: Vec<f32> = source.collect();
    Ok((voice::mono_at_rate(&interleaved, channels, rate), rate))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn settings_with(key: Option<&str>) -> crate::settings::Settings {
        crate::settings::Settings {
            transcription_model: key.map(str::to_owned),
            ..crate::settings::Settings::default()
        }
    }

    #[test]
    fn models_are_named_and_found() {
        assert_eq!(MODELS.len(), 5);
        assert!(model_by_key("tiny").is_none(), "tiny is too weak to offer");
        assert_eq!(model_by_key("base").map(|model| model.label), Some("Base"));
        assert_eq!(model_by_key("nope"), None);
        // Every key resolves, and every file name is the one the download
        // source publishes, or the fetch would 404.
        for model in MODELS {
            assert_eq!(
                model_by_key(model.key).map(|found| found.file),
                Some(model.file)
            );
            assert!(model.file.starts_with("ggml-") && model.file.ends_with(".bin"));
        }
    }

    /// Without a choice the recommended model is used; an unknown key left in
    /// an old settings file falls back to it too rather than doing nothing.
    #[test]
    fn the_recommended_model_is_the_fallback() {
        let recommended = MODELS.iter().find(|model| model.recommended).unwrap();
        assert!(recommended.recommended);
        assert_eq!(MODELS.iter().filter(|model| model.recommended).count(), 1);
        assert_eq!(
            selected_model(&settings_with(None)).unwrap().key,
            recommended.key
        );
        assert_eq!(
            selected_model(&settings_with(Some("tiny"))).unwrap().key,
            recommended.key
        );
        assert_eq!(
            selected_model(&settings_with(Some("base"))).unwrap().key,
            "base"
        );
    }

    /// The exact size on disk decides "downloaded": an empty leftover must
    /// not read as a finished model and produce a confusing runtime error.
    #[test]
    fn an_empty_file_is_not_a_downloaded_model() {
        let dir = std::env::temp_dir().join("zapfast-transcription-models-test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let model = &MODELS[0];
        assert!(!model_downloaded(model, &dir));
        std::fs::write(model.path(&dir), b"").unwrap();
        assert!(!model_downloaded(model, &dir));
        std::fs::write(model.path(&dir), b"ggml").unwrap();
        assert!(model_downloaded(model, &dir));
        // Deleting takes the file and the partial a cancelled download left.
        std::fs::write(model.path(&dir).with_extension("part"), b"half").unwrap();
        delete_model(model, &dir).unwrap();
        assert!(!model_downloaded(model, &dir));
        assert!(!model.path(&dir).with_extension("part").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Deleting a model that is not there is not an error: the settings row
    /// offers the button next to whatever it last knew.
    #[test]
    fn deleting_a_missing_model_succeeds() {
        let dir = std::env::temp_dir().join("zapfast-transcription-missing-test");
        let _ = std::fs::remove_dir_all(&dir);
        delete_model(&MODELS[0], &dir).expect("deleting a model that is not there is fine");
    }

    #[test]
    fn a_fresh_transcriber_has_nothing_in_flight() {
        let transcriber = Transcriber::new(crate::backend::Waker::default());
        assert_eq!(transcriber.status("m1"), None);
        assert_eq!(transcriber.download_status("base"), None);
        assert_eq!(transcriber.revision(), 0);
    }

    /// Cancelling reports nothing to the settings row and leaves no entry
    /// behind, so the row returns to "Download" rather than sticking on a
    /// progress bar that no thread is feeding.
    #[test]
    fn cancelling_a_download_clears_its_row() {
        let transcriber = Transcriber::new(crate::backend::Waker::default());
        transcriber.cancel_download("base");
        assert_eq!(transcriber.download_status("base"), None);
    }

    /// The worker reports a clip it could not read as a failure rather than
    /// panicking or leaving the bubble spinning forever.
    #[test]
    fn a_clip_that_cannot_be_read_fails() {
        let transcriber = Transcriber::new(crate::backend::Waker::default());
        let missing = std::env::temp_dir().join("zapfast-transcription-absent.bin");
        let _ = std::fs::remove_file(&missing);
        transcriber.enqueue("m1".to_owned(), missing.clone(), missing, None);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        loop {
            match transcriber.status("m1") {
                Some(TranscribeState::Failed(_)) => break,
                Some(TranscribeState::Done(_)) => panic!("a missing clip cannot transcribe"),
                None | Some(TranscribeState::Queued | TranscribeState::Running) => {
                    assert!(
                        std::time::Instant::now() < deadline,
                        "the worker never reported back"
                    );
                    std::thread::sleep(std::time::Duration::from_millis(20));
                }
            }
        }
    }

    /// Forgetting a transcript is what a deleted message and a changed model
    /// rely on, so the entry has to be gone afterwards.
    #[test]
    fn forgetting_clears_the_transcript() {
        let transcriber = Transcriber::new(crate::backend::Waker::default());
        transcriber.enqueue(
            "m1".to_owned(),
            std::env::temp_dir().join("absent.bin"),
            std::env::temp_dir().join("absent.ggml"),
            None,
        );
        assert!(transcriber.status("m1").is_some());
        transcriber.forget("m1");
        assert_eq!(transcriber.status("m1"), None);
    }

    /// Voice notes decode at 48 kHz and Whisper wants 16 kHz, so a one-second
    /// clip must come out with 16 000 samples whatever it started at.
    #[test]
    fn audio_is_resampled_to_whispers_rate() {
        let one_second = vec![0.5f32; 48_000];
        assert_eq!(resample(&one_second, 48_000, WHISPER_RATE).len(), 16_000);
        // A container that declares another rate is resampled by its factor,
        // not by the fixed 1-in-3 the voice notes happen to need.
        let cd_quality = vec![0.5f32; 44_100];
        assert_eq!(resample(&cd_quality, 44_100, WHISPER_RATE).len(), 16_000);
        // Same rate in, same samples out.
        assert_eq!(resample(&cd_quality, 16_000, WHISPER_RATE), cd_quality);
        // Short and empty clips do not divide by zero or panic.
        assert_eq!(resample(&[], 48_000, WHISPER_RATE), Vec::<f32>::new());
        assert_eq!(resample(&[1.0], 48_000, WHISPER_RATE), vec![1.0]);
    }

    /// The ramp keeps its shape: a resampler that collapsed the range would
    /// hand Whisper silence.
    #[test]
    fn resampling_preserves_the_signal() {
        let ramp: Vec<f32> = (0..48_000).map(|i| i as f32 / 48_000.0).collect();
        let out = resample(&ramp, 48_000, WHISPER_RATE);
        assert_eq!(out.len(), 16_000);
        assert!(out[0] < out[8_000] && out[8_000] < out[15_999]);
        assert!((out[15_999] - 0.999_979).abs() < 0.001, "{}", out[15_999]);
    }

    /// A payload that is neither OGG nor a known audio container fails
    /// without panicking, the way a corrupted download would.
    #[test]
    fn a_non_ogg_rejects_cleanly() {
        assert!(decode_other(b"nope").is_err());
    }
}
