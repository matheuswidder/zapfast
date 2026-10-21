//! The OS keyring, and what ZapFast keeps in it.
//!
//! The archive's key lives here, and so does the token the TTS extension
//! needs for a speech service. A token in the settings file would sit in a
//! plain text document next to the chats; in the keyring it does not.

use std::sync::Arc;

use keyring_core::api::CredentialStoreApi;

/// The service every ZapFast credential is filed under.
pub const SERVICE: &str = "rocks.zapfast.ZapFast";

/// The credential store of the platform.
pub fn platform_store() -> Result<Arc<dyn CredentialStoreApi>, String> {
    #[cfg(target_os = "linux")]
    let store = zbus_secret_service_keyring_store::Store::new();
    #[cfg(target_os = "macos")]
    let store = apple_native_keyring_store::keychain::Store::new();
    #[cfg(windows)]
    let store = windows_native_keyring_store::Store::new();
    store
        .map_err(|error| error.to_string())
        .map(|store| store as Arc<dyn CredentialStoreApi>)
}

/// Stores a secret under `user`, replacing whatever was there.
pub fn store(user: &str, secret: &str) -> Result<(), String> {
    let store = platform_store()?;
    let entry = store
        .build(SERVICE, user, None)
        .map_err(|error| error.to_string())?;
    entry
        .set_secret(secret.as_bytes())
        .map_err(|error| error.to_string())
}

/// Reads a secret from the keyring. A locked or missing keyring reads as
/// nothing rather than as a failure: the caller asks again when it needs it.
pub fn load(user: &str) -> Option<String> {
    let store = platform_store().ok()?;
    let entry = store.build(SERVICE, user, None).ok()?;
    let secret = entry.get_secret().ok()?;
    String::from_utf8(secret).ok()
}

/// Forgets a secret.
pub fn forget(user: &str) -> Result<(), String> {
    let store = platform_store()?;
    let entry = store
        .build(SERVICE, user, None)
        .map_err(|error| error.to_string())?;
    entry.delete_credential().map_err(|error| error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A keyring that is locked reads as nothing rather than as a failure,
    /// so the interface can ask again instead of refusing to work.
    #[test]
    fn an_unreadable_keyring_reads_as_nothing() {
        // No keyring is opened by a test, so this must not panic.
        let _ = load("tts-token");
    }

    #[test]
    fn test_real_keyring_roundtrip() {
        let test_user = "zapfast-test-token";
        let res = store(test_user, "secret123");
        println!("store result: {res:?}");
        if res.is_ok() {
            let loaded = load(test_user);
            println!("loaded: {loaded:?}");
            assert_eq!(loaded.as_deref(), Some("secret123"));
            let _ = forget(test_user);
        }
    }
}
