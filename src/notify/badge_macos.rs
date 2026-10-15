//! The macOS badge: the unread count on the Dock tile, as WhatsApp shows it.
//!
//! The Dock tile belongs to the running process, so the label is set through
//! `NSApp.dockTile` on the main thread, where the interface runs, and an empty
//! label clears it. No notification permission is needed for it.

use objc2::MainThreadMarker;
use objc2_app_kit::NSApplication;
use objc2_foundation::NSString;

/// Counts unread chats on the Dock tile.
#[derive(Default)]
pub struct Badge {
    shown: Option<u32>,
}

impl Badge {
    /// Shows the count on the Dock tile, clearing it at zero. A repeat is not
    /// applied, so calling this every frame costs nothing.
    pub fn set(&mut self, count: u32) {
        if self.shown == Some(count) {
            return;
        }
        // AppKit is main-thread only; the interface always calls from there.
        let Some(main) = MainThreadMarker::new() else {
            return;
        };
        let tile = NSApplication::sharedApplication(main).dockTile();
        let label = label(count).map(|label| NSString::from_str(&label));
        tile.setBadgeLabel(label.as_deref());
        self.shown = Some(count);
    }
}

/// The Dock label for a count: none at zero, else the number.
fn label(count: u32) -> Option<String> {
    (count > 0).then(|| count.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_clears_the_label_and_other_counts_show_as_numbers() {
        assert_eq!(label(0), None);
        assert_eq!(label(1).as_deref(), Some("1"));
        assert_eq!(label(1234).as_deref(), Some("1234"));
    }
}
