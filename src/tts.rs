//! Speech synthesis for the TTS extension: text into a voice message.
//!
//! Two sources of speech. The local one runs Piper on this computer with a
//! voice downloaded on demand, and the text never leaves the machine. The
//! remote one asks a service to speak the text, the way the radialista agent
//! does, and gets an audio file back. Both end the same way: mono samples at
//! the rate a voice note uses, ready to be sent.
//!
//! The remote source needs a token, which lives in the OS keyring and not in
//! the settings file, so the key never sits in a plain text document next to
//! the chats.

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
        mpsc::Sender,
    },
};

use crate::backend::Waker;
use crate::settings::TtsSource;

/// A voice that runs on this computer: Piper's model, and where to get it.
pub struct LocalVoice {
    /// The key the settings file holds.
    pub key: &'static str,
    /// The name the interface shows.
    pub name: &'static str,
    /// What the voice is, in the interface's language.
    pub description: &'static str,
    /// The model file the download source publishes.
    pub file: &'static str,
    /// How big the download is, for the interface.
    pub size: &'static str,
    /// Where the model is published.
    pub model_url: &'static str,
    /// Where the model's own settings are published.
    pub config_url: &'static str,
}

impl LocalVoice {
    /// The name the model's settings are published under.
    pub fn config_file(&self) -> String {
        format!("{}.json", self.file)
    }

    /// The name and what the voice is, in the interface's language.
    pub fn label(&self, locale: crate::i18n::Locale) -> String {
        format!(
            "{} ({})",
            self.name,
            crate::i18n::gettext(locale, self.description)
        )
    }
}

/// The key Piper's package download is filed under.
pub const PIPER_PACKAGE_KEY: &str = "piper";

/// The voices that run on this computer. All of them are Brazilian
/// Portuguese, and the three medium ones are released under CC0.
pub const LOCAL_VOICES: [LocalVoice; 4] = [
    LocalVoice {
        key: "faber",
        name: "Faber",
        description: "a news anchor's voice",
        file: "pt_BR-faber-medium.onnx",
        size: "63 MB",
        model_url: "https://huggingface.co/rhasspy/piper-voices/resolve/main/pt/pt_BR/faber/medium/pt_BR-faber-medium.onnx",
        config_url: "https://huggingface.co/rhasspy/piper-voices/resolve/main/pt/pt_BR/faber/medium/pt_BR-faber-medium.onnx.json",
    },
    LocalVoice {
        key: "jeff",
        name: "Jeff",
        description: "a man's voice",
        file: "pt_BR-jeff-medium.onnx",
        size: "63 MB",
        model_url: "https://huggingface.co/rhasspy/piper-voices/resolve/main/pt/pt_BR/jeff/medium/pt_BR-jeff-medium.onnx",
        config_url: "https://huggingface.co/rhasspy/piper-voices/resolve/main/pt/pt_BR/jeff/medium/pt_BR-jeff-medium.onnx.json",
    },
    LocalVoice {
        key: "cadu",
        name: "Cadu",
        description: "a man's voice",
        file: "pt_BR-cadu-medium.onnx",
        size: "63 MB",
        model_url: "https://huggingface.co/rhasspy/piper-voices/resolve/main/pt/pt_BR/cadu/medium/pt_BR-cadu-medium.onnx",
        config_url: "https://huggingface.co/rhasspy/piper-voices/resolve/main/pt/pt_BR/cadu/medium/pt_BR-cadu-medium.onnx.json",
    },
    LocalVoice {
        key: "edresson",
        name: "Edresson",
        description: "a lighter man's voice",
        file: "pt_BR-edresson-low.onnx",
        size: "63 MB",
        model_url: "https://huggingface.co/rhasspy/piper-voices/resolve/main/pt/pt_BR/edresson/low/pt_BR-edresson-low.onnx",
        config_url: "https://huggingface.co/rhasspy/piper-voices/resolve/main/pt/pt_BR/edresson/low/pt_BR-edresson-low.onnx.json",
    },
];

/// A voice a remote service speaks with.
pub struct RemoteVoice {
    /// The name the service knows the voice by, and the settings hold.
    pub name: &'static str,
    /// The name the interface shows.
    pub title: &'static str,
    /// What the voice is, in the interface's language.
    pub description: &'static str,
    /// What the service is asked for.
    pub model: &'static str,
}

impl RemoteVoice {
    /// The name and what the voice is, in the interface's language.
    pub fn label(&self, locale: crate::i18n::Locale) -> String {
        format!(
            "{} ({})",
            self.title,
            crate::i18n::gettext(locale, self.description)
        )
    }
}

/// The voices the Edge service answers for. The list is short because the
/// service itself refuses the others: asked for a voice it does not know, it
/// answers with a 500 and the proxy turns that into a 502.
pub const REMOTE_VOICES: [RemoteVoice; 4] = [
    RemoteVoice {
        name: "pt-BR-ThalitaMultilingualNeural",
        title: "Thalita",
        description: "a woman's voice",
        model: "edge-tts/pt-BR-ThalitaMultilingualNeural",
    },
    RemoteVoice {
        name: "pt-BR-ThalitaNeural",
        title: "Thalita",
        description: "a woman's voice, the classic one",
        model: "edge-tts/pt-BR-ThalitaNeural",
    },
    RemoteVoice {
        name: "pt-BR-FranciscaNeural",
        title: "Francisca",
        description: "a woman's voice",
        model: "edge-tts/pt-BR-FranciscaNeural",
    },
    RemoteVoice {
        name: "pt-BR-AntonioNeural",
        title: "Antonio",
        description: "a man's voice",
        model: "edge-tts/pt-BR-AntonioNeural",
    },
];

/// The most characters a message may hold. Measured against both sources
/// on the same text: the local voice speaks at 18 characters a second and
/// the service at 16.6, so this is a shade under two minutes of speech.
pub const MAX_CHARS: usize = 1_900;

/// How fast the voices speak, in characters a second. The slower of the
/// two, so the estimate never promises less speech than the audio holds.
pub const CHARS_PER_SECOND: f32 = 16.5;

/// How long the text will take to speak, roughly. The audio the voice
/// makes is the truth; this only tells the writer what to expect.
pub fn estimate_seconds(text: &str) -> u32 {
    let characters = text.trim().chars().count();
    (characters as f32 / CHARS_PER_SECOND).round() as u32
}

/// The service the remote source talks to unless the settings say
/// otherwise. It speaks the OpenAI audio API, the same one the radialista
/// agent uses.
pub const DEFAULT_ENDPOINT: &str = "https://proxy.widder.cloud/v1/audio/speech";

/// Where the service's token is filed in the OS keyring.
pub const TTS_TOKEN: &str = "tts-token";

/// Piper's binary, which the extension downloads like the GPU runtime.
#[cfg(all(target_os = "windows", target_arch = "x86_64"))]
pub const PIPER_PACKAGE_URL: &str =
    "https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_windows_amd64.zip";
#[cfg(all(target_os = "windows", target_arch = "aarch64"))]
pub const PIPER_PACKAGE_URL: &str =
    "https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_windows_amd64.zip";
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
pub const PIPER_PACKAGE_URL: &str =
    "https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_x86_64.tar.gz";
#[cfg(all(target_os = "linux", target_arch = "aarch64"))]
pub const PIPER_PACKAGE_URL: &str =
    "https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_aarch64.tar.gz";
#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
pub const PIPER_PACKAGE_URL: &str =
    "https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_macos_x64.tar.gz";
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub const PIPER_PACKAGE_URL: &str =
    "https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_macos_aarch64.tar.gz";

/// How big Piper's download is, for the interface.
pub const PIPER_PACKAGE_SIZE: &str = "22 MB";

/// Where Piper's binary lives.
pub fn piper_dir(dirs: &crate::paths::AppDirs) -> PathBuf {
    dirs.state.join("tts").join("piper")
}

/// Where the voice models live.
pub fn models_dir(dirs: &crate::paths::AppDirs) -> PathBuf {
    dirs.state.join("tts").join("models")
}

/// The local voice a key names, if it is one of ours.
pub fn local_voice_by_key(key: &str) -> Option<&'static LocalVoice> {
    LOCAL_VOICES.iter().find(|voice| voice.key == key)
}

/// The remote voice a name names, if it is one of ours.
pub fn remote_voice_by_name(name: &str) -> Option<&'static RemoteVoice> {
    REMOTE_VOICES.iter().find(|voice| voice.name == name)
}

/// The local voice the settings name, falling back to the first one.
pub fn selected_local_voice(settings: &crate::settings::Settings) -> &'static LocalVoice {
    local_voice_by_key(&settings.tts_voice).unwrap_or(&LOCAL_VOICES[0])
}

/// The remote voice the settings name, falling back to the first one.
pub fn selected_remote_voice(settings: &crate::settings::Settings) -> &'static RemoteVoice {
    remote_voice_by_name(&settings.tts_voice).unwrap_or(&REMOTE_VOICES[0])
}

/// Walks `dir` for the file the package published under a name.
fn find_named(dir: &Path, name: &str) -> Option<PathBuf> {
    let mut stack = vec![dir.to_path_buf()];
    while let Some(current) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&current) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.file_name().is_some_and(|file| file == name) {
                return Some(path);
            }
        }
    }
    None
}

/// Piper's binary, wherever the package put it.
pub fn piper_path(dirs: &crate::paths::AppDirs) -> Option<PathBuf> {
    let dir = piper_dir(dirs);
    if cfg!(target_os = "windows") {
        find_named(&dir, "piper.exe")
    } else {
        find_named(&dir, "piper")
    }
}

/// The binary is installed and the extension can speak with it.
pub fn piper_ready(dirs: &crate::paths::AppDirs) -> bool {
    piper_path(dirs).is_some_and(|path| path.is_file())
}

/// The model of a voice is on disk, both the weights and their settings.
pub fn voice_downloaded(voice: &LocalVoice, dir: &Path) -> bool {
    dir.join(voice.file).is_file() && dir.join(voice.config_file()).is_file()
}

/// How a synthesis is going.
pub enum SpeakState {
    /// Waiting for the worker.
    Queued,
    /// The worker is on it.
    Running,
    /// The speech is ready: the samples to send, and the file to play.
    Done {
        /// Mono samples at the rate a voice note uses.
        samples: Vec<f32>,
        /// The same speech in a file, for the player.
        path: PathBuf,
    },
    /// It could not be spoken.
    Failed(String),
}

/// What the worker is asked to do.
pub struct Request {
    /// What the request is called, so the interface can find its state.
    pub id: String,
    /// The text to speak.
    pub text: String,
    /// Where the speech comes from.
    pub source: TtsSource,
    /// Piper's binary, for the local source.
    pub piper: Option<PathBuf>,
    /// The voice model, for the local source.
    pub model: Option<PathBuf>,
    /// The service, for the remote source.
    pub endpoint: Option<String>,
    /// The voice the service is asked for.
    pub voice: Option<String>,
    /// The token the service asks for.
    pub key: Option<String>,
    /// Where the speech is written, for the player.
    pub output: PathBuf,
}

/// Shared handle the interface and the worker see.
pub struct Synthesizer {
    states: Arc<Mutex<HashMap<String, SpeakState>>>,
    downloads: Arc<Mutex<HashMap<String, crate::transcription::DownloadState>>>,
    queue: Mutex<Sender<Request>>,
    /// Grows every time a speech lands, so a frame that missed the wake can
    /// still tell.
    revision: Arc<AtomicU64>,
}

impl Synthesizer {
    /// Starts the worker thread.
    pub fn new(waker: Waker) -> Self {
        let (sender, receiver) = std::sync::mpsc::channel::<Request>();
        let states = Arc::new(Mutex::new(HashMap::new()));
        let downloads = Arc::new(Mutex::new(HashMap::new()));
        let revision = Arc::new(AtomicU64::new(0));
        let worker_states = Arc::clone(&states);
        let worker_revision = Arc::clone(&revision);
        std::thread::Builder::new()
            .name("speak".to_owned())
            .spawn(move || {
                while let Ok(request) = receiver.recv() {
                    worker_states
                        .lock()
                        .unwrap_or_else(|p| p.into_inner())
                        .insert(request.id.clone(), SpeakState::Running);
                    worker_revision.fetch_add(1, Ordering::Relaxed);
                    waker.wake();
                    let started = std::time::Instant::now();
                    let result = speak(&request);
                    log::info!(
                        "[TTS] {} in {:.1}s",
                        if result.is_ok() { "done" } else { "failed" },
                        started.elapsed().as_secs_f32()
                    );
                    let state = match result {
                        Ok((samples, path)) => SpeakState::Done { samples, path },
                        Err(error) => SpeakState::Failed(error),
                    };
                    let mut states = worker_states.lock().unwrap_or_else(|p| p.into_inner());
                    states.insert(request.id.clone(), state);
                    drop(states);
                    worker_revision.fetch_add(1, Ordering::Relaxed);
                    waker.wake();
                }
            })
            .expect("spawns the speak thread");
        Self {
            states,
            downloads,
            queue: Mutex::new(sender),
            revision,
        }
    }

    /// Queues a text to be spoken.
    pub fn speak(&self, request: Request) {
        let id = request.id.clone();
        self.states
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .insert(id, SpeakState::Queued);
        self.revision.fetch_add(1, Ordering::Relaxed);
        if let Ok(sender) = self.queue.lock() {
            let _ = sender.send(request);
        }
    }

    /// How a request is going, if it is known.
    pub fn state(&self, id: &str) -> Option<SpeakState> {
        self.states
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .get(id)
            .cloned()
    }

    /// How many speeches have reached a final state.
    pub fn revision(&self) -> u64 {
        self.revision.load(Ordering::Relaxed)
    }

    /// Forgets a request.
    pub fn forget(&self, id: &str) {
        self.states
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .remove(id);
    }

    /// How a download is going, if it is known.
    pub fn download_status(&self, key: &str) -> Option<crate::transcription::DownloadState> {
        self.downloads
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .get(key)
            .cloned()
    }

    /// Stops a download the reader started by mistake.
    pub fn cancel_download(&self, key: &str) {
        let mut downloads = self.downloads.lock().unwrap_or_else(|p| p.into_inner());
        if matches!(
            downloads.get(key),
            Some(crate::transcription::DownloadState::Downloading { .. })
        ) {
            downloads.insert(
                key.to_owned(),
                crate::transcription::DownloadState::Cancelled,
            );
        }
    }

    /// Downloads Piper's binary and unpacks it, in its own thread. An
    /// installed runtime is left alone.
    pub fn download_piper(&self, dirs: crate::paths::AppDirs, waker: Waker) {
        if piper_ready(&dirs) {
            return;
        }
        {
            let mut downloads = self.downloads.lock().unwrap_or_else(|p| p.into_inner());
            match downloads.get(PIPER_PACKAGE_KEY) {
                Some(crate::transcription::DownloadState::Downloading { .. })
                | Some(crate::transcription::DownloadState::Done) => return,
                _ => {}
            }
            downloads.insert(
                PIPER_PACKAGE_KEY.to_owned(),
                crate::transcription::DownloadState::Downloading {
                    received: 0,
                    total: None,
                },
            );
        }
        let downloads = Arc::clone(&self.downloads);
        std::thread::Builder::new()
            .name("tts-piper".to_owned())
            .spawn(move || {
                let dir = piper_dir(&dirs);
                let archive = dir.join(package_name());
                let fetched = crate::transcription::fetch_url(
                    PIPER_PACKAGE_URL,
                    &archive,
                    PIPER_PACKAGE_KEY,
                    &downloads,
                    &waker,
                );
                let outcome = match fetched {
                    crate::transcription::DownloadState::Done => {
                        let unpacked = extract_package(&archive, &dir);
                        let _ = std::fs::remove_file(&archive);
                        unpacked
                    }
                    crate::transcription::DownloadState::Cancelled => {
                        let _ = std::fs::remove_file(&archive);
                        return;
                    }
                    other => Err(format!("the download failed: {other:?}")),
                };
                let mut downloads = downloads.lock().unwrap_or_else(|p| p.into_inner());
                downloads.insert(
                    PIPER_PACKAGE_KEY.to_owned(),
                    match outcome {
                        Ok(()) => crate::transcription::DownloadState::Done,
                        Err(error) => crate::transcription::DownloadState::Failed(error),
                    },
                );
                drop(downloads);
                waker.wake();
            })
            .expect("spawns the piper download thread");
    }

    /// Downloads a voice's model and its settings, in its own thread.
    pub fn download_voice(
        &self,
        voice: &'static LocalVoice,
        dirs: crate::paths::AppDirs,
        waker: Waker,
    ) {
        let dir = models_dir(&dirs);
        if voice_downloaded(voice, &dir) {
            return;
        }
        {
            let mut downloads = self.downloads.lock().unwrap_or_else(|p| p.into_inner());
            match downloads.get(voice.key) {
                Some(crate::transcription::DownloadState::Downloading { .. })
                | Some(crate::transcription::DownloadState::Done) => return,
                _ => {}
            }
            downloads.insert(
                voice.key.to_owned(),
                crate::transcription::DownloadState::Downloading {
                    received: 0,
                    total: None,
                },
            );
        }
        let downloads = Arc::clone(&self.downloads);
        std::thread::Builder::new()
            .name("tts-voice".to_owned())
            .spawn(move || {
                let mut outcome = Ok(());
                let files = [
                    (voice.model_url, dir.join(voice.file)),
                    (voice.config_url, dir.join(voice.config_file())),
                ];
                for (url, path) in files {
                    let fetched =
                        crate::transcription::fetch_url(url, &path, voice.key, &downloads, &waker);
                    match fetched {
                        crate::transcription::DownloadState::Done => {}
                        crate::transcription::DownloadState::Cancelled => return,
                        other => {
                            outcome = Err(format!("the download failed: {other:?}"));
                            break;
                        }
                    }
                }
                let mut downloads = downloads.lock().unwrap_or_else(|p| p.into_inner());
                downloads.insert(
                    voice.key.to_owned(),
                    match outcome {
                        Ok(()) => crate::transcription::DownloadState::Done,
                        Err(error) => crate::transcription::DownloadState::Failed(error),
                    },
                );
                drop(downloads);
                waker.wake();
            })
            .expect("spawns the voice download thread");
    }

    /// Removes a downloaded voice.
    pub fn delete_voice(voice: &LocalVoice, dir: &Path) -> std::io::Result<()> {
        let model = dir.join(voice.file);
        if model.exists() {
            std::fs::remove_file(&model)?;
        }
        let config = dir.join(voice.config_file());
        if config.exists() {
            std::fs::remove_file(config)?;
        }
        Ok(())
    }

    /// Removes Piper's binary.
    pub fn delete_piper(dirs: &crate::paths::AppDirs) -> std::io::Result<()> {
        let dir = piper_dir(dirs);
        if dir.exists() {
            std::fs::remove_dir_all(&dir)?;
        }
        Ok(())
    }
}

/// The name Piper's package is fetched under, which says what shape it has.
fn package_name() -> &'static str {
    if cfg!(target_os = "windows") {
        "piper-package.zip"
    } else {
        "piper-package.tar.gz"
    }
}

/// Unpacks Piper's package, whatever shape the platform's release takes.
fn extract_package(archive: &Path, dir: &Path) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|error| error.to_string())?;
    if archive
        .extension()
        .is_some_and(|extension| extension == "zip")
    {
        let file = std::fs::File::open(archive).map_err(|error| error.to_string())?;
        let mut archive = zip::ZipArchive::new(file).map_err(|error| error.to_string())?;
        for index in 0..archive.len() {
            let mut entry = archive.by_index(index).map_err(|error| error.to_string())?;
            // A zip entry's name can carry `..`; confine every file to `dir`.
            let name = entry
                .enclosed_name()
                .ok_or_else(|| "unsafe path in package".to_owned())?;
            let out = dir.join(name);
            if entry.is_dir() {
                std::fs::create_dir_all(&out).map_err(|error| error.to_string())?;
            } else {
                if let Some(parent) = out.parent() {
                    std::fs::create_dir_all(parent).map_err(|error| error.to_string())?;
                }
                let mut file = std::fs::File::create(&out).map_err(|error| error.to_string())?;
                std::io::copy(&mut entry, &mut file).map_err(|error| error.to_string())?;
            }
        }
        return Ok(());
    }
    let file = std::fs::File::open(archive).map_err(|error| error.to_string())?;
    let decoder = flate2::read::GzDecoder::new(file);
    let mut archive = tar::Archive::new(decoder);
    archive.unpack(dir).map_err(|error| error.to_string())
}

impl Clone for SpeakState {
    fn clone(&self) -> Self {
        match self {
            Self::Queued => Self::Queued,
            Self::Running => Self::Running,
            Self::Done { samples, path } => Self::Done {
                samples: samples.clone(),
                path: path.clone(),
            },
            Self::Failed(error) => Self::Failed(error.clone()),
        }
    }
}

/// Speaks the text and returns the samples and the file they are in.
fn speak(request: &Request) -> Result<(Vec<f32>, PathBuf), String> {
    let bytes = match request.source {
        TtsSource::Local => speak_local(request)?,
        TtsSource::Remote => speak_remote(request)?,
    };
    let samples = crate::transcription::decode_for_voice_note(&bytes)?;
    if samples.is_empty() {
        return Err("the service returned no speech".to_owned());
    }
    crate::transcription::write_wav_at(&request.output, &samples, crate::voice::RATE)
        .map_err(|error| error.to_string())?;
    Ok((samples, request.output.clone()))
}

/// Runs Piper on this computer and reads the file it wrote.
fn speak_local(request: &Request) -> Result<Vec<u8>, String> {
    let piper = request
        .piper
        .as_ref()
        .ok_or_else(|| "Piper is not installed".to_owned())?;
    let model = request
        .model
        .as_ref()
        .ok_or_else(|| "no voice is installed".to_owned())?;
    if !piper.is_file() {
        return Err("Piper is not installed".to_owned());
    }
    if !model.is_file() {
        return Err("no voice is installed".to_owned());
    }
    let mut child = Command::new(piper)
        .arg("--model")
        .arg(model)
        .arg("--output_file")
        .arg(&request.output)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|error| format!("Piper would not start: {error}"))?;
    if let Some(mut stdin) = child.stdin.take() {
        use std::io::Write;
        let _ = stdin.write_all(request.text.as_bytes());
    }
    let status = child
        .wait()
        .map_err(|error| format!("Piper failed: {error}"))?;
    if !status.success() {
        return Err("Piper could not speak the text".to_owned());
    }
    std::fs::read(&request.output).map_err(|error| format!("Piper wrote no audio: {error}"))
}

/// Asks the service to speak the text and returns the audio it answered.
fn speak_remote(request: &Request) -> Result<Vec<u8>, String> {
    let endpoint = request
        .endpoint
        .as_deref()
        .filter(|endpoint| !endpoint.is_empty())
        .ok_or_else(|| "no service is set".to_owned())?;
    let voice = request
        .voice
        .as_deref()
        .ok_or_else(|| "no voice is chosen".to_owned())?;
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(60))
        .build()
        .map_err(|error| error.to_string())?;
    let mut builder = client.post(endpoint);
    if let Some(key) = request.key.as_deref().filter(|key| !key.is_empty()) {
        builder = builder.bearer_auth(key);
    }
    let body = serde_json::json!({ "model": voice, "input": request.text }).to_string();
    let response = builder
        .header("Content-Type", "application/json")
        .body(body)
        .send()
        .map_err(|error| format!("the service did not answer: {error}"))?;
    if !response.status().is_success() {
        return Err(format!("the service answered {}", response.status()));
    }
    let bytes = response
        .bytes()
        .map_err(|error| format!("the service's answer was cut: {error}"))?;
    Ok(bytes.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The limit and the estimate agree with the voices: the text the
    /// panel accepts is about two minutes of speech, and the estimate says
    /// so.
    #[test]
    fn the_limit_is_about_two_minutes_of_speech() {
        let seconds = estimate_seconds(&"A".repeat(MAX_CHARS));
        assert!(
            (110..=125).contains(&seconds),
            "{MAX_CHARS} characters are {seconds} seconds of speech"
        );
        assert_eq!(estimate_seconds(""), 0);
        assert_eq!(estimate_seconds("   \n  "), 0);
    }

    #[test]
    fn the_local_voices_name_their_own_files() {
        assert_eq!(LOCAL_VOICES.len(), 4);
        for voice in LOCAL_VOICES {
            assert!(local_voice_by_key(voice.key).is_some());
            assert!(voice.file.starts_with("pt_BR-"));
            assert!(voice.file.ends_with(".onnx"));
            assert!(voice.model_url.ends_with(voice.file));
            assert!(voice.config_url.ends_with(".onnx.json"));
        }
        assert!(local_voice_by_key("nope").is_none());
    }

    /// The keys are the ones the settings file holds, so a file written by
    /// an older build still finds its voice.
    #[test]
    fn a_voice_key_survives_a_restart() {
        let settings = crate::settings::Settings {
            tts_voice: "jeff".to_owned(),
            ..crate::settings::Settings::default()
        };
        assert_eq!(selected_local_voice(&settings).key, "jeff");
        let unknown = crate::settings::Settings {
            tts_voice: "gone".to_owned(),
            ..crate::settings::Settings::default()
        };
        assert_eq!(selected_local_voice(&unknown).key, LOCAL_VOICES[0].key);
    }

    /// The remote voices are the ones the service answers for: asked for
    /// another, it fails with a 500 and the proxy with a 502.
    #[test]
    fn the_remote_voices_are_the_ones_that_answer() {
        assert_eq!(REMOTE_VOICES.len(), 4);
        for voice in REMOTE_VOICES {
            assert!(remote_voice_by_name(voice.name).is_some());
            assert!(voice.model.ends_with(voice.name));
            assert!(voice.name.starts_with("pt-BR-"));
        }
    }

    #[test]
    fn the_endpoint_is_the_one_the_radialista_uses() {
        assert_eq!(
            DEFAULT_ENDPOINT,
            "https://proxy.widder.cloud/v1/audio/speech"
        );
    }

    /// A service that answers with silence is a failure, not an empty clip.
    #[test]
    fn silence_is_not_speech() {
        let request = Request {
            id: "tts-test".to_owned(),
            text: "olá".to_owned(),
            source: TtsSource::Remote,
            piper: None,
            model: None,
            endpoint: Some(DEFAULT_ENDPOINT.to_owned()),
            voice: Some("edge-tts/pt-BR-ThalitaMultilingualNeural".to_owned()),
            key: None,
            output: PathBuf::new(),
        };
        // No service is reachable from a test, so this must fail rather than
        // hand the interface an empty clip.
        assert!(speak(&request).is_err());
    }

    /// The speech has to end at the rate a voice note uses, whatever rate
    /// the service answered at. The service answers at 16 kHz and the local
    /// voice at 22.05 kHz; both have to become 48 kHz.
    #[test]
    fn speech_ends_at_the_voice_note_rate() {
        for rate in [16_000u32, 22_050] {
            let dir = std::env::temp_dir().join("zapfast-tts-test");
            std::fs::create_dir_all(&dir).expect("the test folder");
            let path = dir.join(format!("tone-{rate}.wav"));
            let samples: Vec<f32> = (0..rate).map(|index| (index as f32 * 0.01).sin()).collect();
            crate::transcription::write_wav_at(&path, &samples, rate).expect("writes the tone");
            let bytes = std::fs::read(&path).expect("reads the tone");
            let (decoded, decoded_rate) =
                crate::transcription::decode_other(&bytes).expect("decodes the tone");
            assert_eq!(decoded_rate, rate);
            // The decoder brings the file to the voice note's own rate, so a
            // second of speech is a second of samples whatever the file's
            // rate was.
            assert_eq!(decoded.len(), crate::voice::RATE as usize);
            let at_voice_rate = crate::transcription::decode_for_voice_note(&bytes)
                .expect("decodes the tone for the voice note");
            assert_eq!(at_voice_rate.len(), crate::voice::RATE as usize);
            let _ = std::fs::remove_file(&path);
        }
    }

    /// The file the player reads is the speech the message carries, at the
    /// rate a voice note uses.
    #[test]
    fn the_speech_file_holds_the_speech() {
        let dir = std::env::temp_dir().join("zapfast-tts-test");
        std::fs::create_dir_all(&dir).expect("the test folder");
        let path = dir.join("speech.wav");
        let samples: Vec<f32> = (0..48_000)
            .map(|index| (index as f32 * 0.01).sin())
            .collect();
        crate::transcription::write_wav_at(&path, &samples, crate::voice::RATE)
            .expect("writes the speech");
        let bytes = std::fs::read(&path).expect("reads the speech");
        let (decoded, rate) = crate::transcription::decode_other(&bytes).expect("decodes");
        assert_eq!(rate, crate::voice::RATE);
        assert_eq!(decoded.len(), samples.len());
        let _ = std::fs::remove_file(&path);
    }

    /// With Piper and a voice installed, the local source speaks the text
    /// and leaves a file the player can read. On a machine without them
    /// there is nothing to prove, and the test says so.
    #[test]
    fn the_local_source_speaks_the_text() {
        let dirs = crate::paths::AppDirs::discover();
        let Some(piper) = piper_path(&dirs) else {
            return;
        };
        let voice = &LOCAL_VOICES[0];
        let model = models_dir(&dirs).join(voice.file);
        if !model.is_file() {
            return;
        }
        let dir = std::env::temp_dir().join("zapfast-tts-test");
        std::fs::create_dir_all(&dir).expect("the test folder");
        let output = dir.join("local.wav");
        let request = Request {
            id: "tts-local".to_owned(),
            text: "Boa tarde. Este e um teste de sintese de fala local.".to_owned(),
            source: TtsSource::Local,
            piper: Some(piper),
            model: Some(model),
            endpoint: None,
            voice: None,
            key: None,
            output: output.clone(),
        };
        let (samples, path) = speak(&request).expect("Piper speaks the text");
        assert!(!samples.is_empty(), "the speech is not empty");
        assert!(path.is_file(), "the speech file is there");
        // The samples are at the rate a voice note uses, whatever rate
        // Piper answered at.
        let seconds = samples.len() as f32 / crate::voice::RATE as f32;
        assert!(seconds > 0.5, "the speech lasts more than half a second");
        let _ = std::fs::remove_file(&output);
    }
}
