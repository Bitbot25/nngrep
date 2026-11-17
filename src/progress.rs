use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use std::sync::OnceLock;
use std::time::Duration;

/// Returns the progress display shared by all bars that may be visible at the
/// same time.
pub fn bars() -> &'static MultiProgress {
    static BARS: OnceLock<MultiProgress> = OnceLock::new();
    BARS.get_or_init(MultiProgress::new)
}

/// Returns a ticking spinner followed by the message template.
pub fn spinner(template: &str) -> ProgressBar {
    let pb = bars().add(ProgressBar::new_spinner().with_style(
        ProgressStyle::with_template(&format!("{{spinner:.green}} {template}")).unwrap(),
    ));
    pb.enable_steady_tick(Duration::from_millis(100));
    pb
}
