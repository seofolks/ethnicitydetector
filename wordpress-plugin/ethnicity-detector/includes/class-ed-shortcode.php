<?php

if (!defined('ABSPATH')) {
    exit;
}

class ED_Shortcode
{
    public static function init(): void
    {
        add_shortcode('ethnicity_detector', [self::class, 'render']);
        add_action('wp_enqueue_scripts', [self::class, 'register_assets']);
    }

    public static function register_assets(): void
    {
        wp_register_style(
            'ethnicity-detector',
            ED_PLUGIN_URL . 'assets/css/detector.css',
            [],
            ED_PLUGIN_VERSION
        );
        wp_register_script(
            'ethnicity-detector',
            ED_PLUGIN_URL . 'assets/js/detector.js',
            [],
            ED_PLUGIN_VERSION,
            true
        );
    }

    public static function render($atts = []): string
    {
        wp_enqueue_style('ethnicity-detector');
        wp_enqueue_script('ethnicity-detector');

        $settings = ED_Settings::get();
        $config = [
            'restUrl' => esc_url_raw(rest_url('ethnicity-detector/v1/analyze')),
            'nonce' => wp_create_nonce('wp_rest'),
            'hasApi' => !empty($settings['api_url']),
        ];
        wp_add_inline_script(
            'ethnicity-detector',
            'window.ED_DETECTOR = ' . wp_json_encode($config) . ';',
            'before'
        );

        $title = esc_html($settings['title']);
        $subtitle = esc_html($settings['subtitle']);

        ob_start();
        ?>
        <section class="ed-tool" data-ed-root>
            <header class="ed-tool__header">
                <p class="ed-tool__brand">What Is My Ethnicity</p>
                <h2 class="ed-tool__title"><?php echo $title; ?></h2>
                <p class="ed-tool__subtitle"><?php echo $subtitle; ?></p>
            </header>

            <?php if (empty($settings['api_url'])) : ?>
                <div class="ed-tool__notice" role="status">
                    Configure the DeepFace API URL in <strong>Settings → Ethnicity Detector</strong> to enable analysis.
                </div>
            <?php endif; ?>

            <div class="ed-tool__modes" role="tablist" aria-label="Input mode">
                <button type="button" class="ed-tool__mode is-active" data-ed-mode="upload" role="tab" aria-selected="true">Upload photo</button>
                <button type="button" class="ed-tool__mode" data-ed-mode="webcam" role="tab" aria-selected="false">Use webcam</button>
            </div>

            <div class="ed-tool__panel" data-ed-panel="upload">
                <label class="ed-drop" data-ed-drop>
                    <input type="file" accept="image/jpeg,image/png,image/webp" data-ed-file hidden />
                    <span class="ed-drop__title">Drop an image here</span>
                    <span class="ed-drop__hint">or click to browse · JPG, PNG, WEBP</span>
                </label>
            </div>

            <div class="ed-tool__panel" data-ed-panel="webcam" hidden>
                <div class="ed-webcam">
                    <video data-ed-video playsinline autoplay muted></video>
                    <canvas data-ed-canvas hidden></canvas>
                    <div class="ed-webcam__actions">
                        <button type="button" class="ed-btn ed-btn--ghost" data-ed-start-cam>Start camera</button>
                        <button type="button" class="ed-btn" data-ed-capture disabled>Capture photo</button>
                    </div>
                </div>
            </div>

            <div class="ed-preview" data-ed-preview hidden>
                <img data-ed-preview-img alt="Selected photo preview" />
                <button type="button" class="ed-btn ed-btn--primary" data-ed-analyze disabled>Analyze</button>
            </div>

            <div class="ed-status" data-ed-status hidden role="status"></div>
            <div class="ed-results" data-ed-results hidden></div>

            <p class="ed-tool__disclaimer">
                Results are AI estimates for education and entertainment only. Images are processed for analysis and not stored by this plugin.
            </p>
        </section>
        <?php
        return (string) ob_get_clean();
    }
}
