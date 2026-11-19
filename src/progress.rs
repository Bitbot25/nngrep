use indicatif::{HumanDuration, MultiProgress, ProgressBar, ProgressState, ProgressStyle};
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

/// Returns a progress bar style for the template, which may also use
/// `{avg_per_sec}` and `{avg_eta}`. These use the average rate since the bar
/// started rather than the recent rate.
pub fn bar_style(template: &str) -> ProgressStyle {
    ProgressStyle::with_template(template)
        .unwrap()
        .with_key(
            "avg_per_sec",
            |s: &ProgressState, w: &mut dyn std::fmt::Write| {
                write!(w, "{:.1}", average_rate(s)).unwrap()
            },
        )
        .with_key(
            "avg_eta",
            |s: &ProgressState, w: &mut dyn std::fmt::Write| {
                let rate = average_rate(s);
                if rate > 0.0 {
                    let left = s.len().unwrap_or(0).saturating_sub(s.pos()) as f64 / rate;
                    write!(w, "{}", HumanDuration(Duration::from_secs_f64(left))).unwrap()
                } else {
                    write!(w, "?").unwrap()
                }
            },
        )
        .progress_chars("=> ")
}

/// Returns the average number of steps per second since the bar started.
fn average_rate(state: &ProgressState) -> f64 {
    let secs = state.elapsed().as_secs_f64();
    if secs > 0.0 {
        state.pos() as f64 / secs
    } else {
        0.0
    }
}
