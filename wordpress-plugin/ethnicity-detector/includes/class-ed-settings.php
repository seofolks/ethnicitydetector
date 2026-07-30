<?php

if (!defined('ABSPATH')) {
    exit;
}

class ED_Settings
{
    public const OPTION_KEY = 'ed_settings';

    public static function init(): void
    {
        add_action('admin_menu', [self::class, 'add_menu']);
        add_action('admin_init', [self::class, 'register_settings']);
    }

    public static function defaults(): array
    {
        return [
            'api_url' => '',
            'api_key' => '',
            'title' => 'What Is My Ethnicity?',
            'subtitle' => 'Upload a photo or use your webcam for AI ethnicity and emotion insights.',
        ];
    }

    public static function get(): array
    {
        return wp_parse_args(get_option(self::OPTION_KEY, []), self::defaults());
    }

    public static function add_menu(): void
    {
        add_options_page(
            'Ethnicity Detector',
            'Ethnicity Detector',
            'manage_options',
            'ethnicity-detector',
            [self::class, 'render_page']
        );
    }

    public static function register_settings(): void
    {
        register_setting('ed_settings_group', self::OPTION_KEY, [
            'type' => 'array',
            'sanitize_callback' => [self::class, 'sanitize'],
            'default' => self::defaults(),
        ]);
    }

    public static function sanitize($input): array
    {
        $input = is_array($input) ? $input : [];
        $out = self::defaults();
        $out['api_url'] = isset($input['api_url']) ? esc_url_raw(trim($input['api_url'])) : '';
        $out['api_key'] = isset($input['api_key']) ? sanitize_text_field($input['api_key']) : '';
        $out['title'] = isset($input['title']) ? sanitize_text_field($input['title']) : $out['title'];
        $out['subtitle'] = isset($input['subtitle']) ? sanitize_text_field($input['subtitle']) : $out['subtitle'];
        return $out;
    }

    public static function render_page(): void
    {
        if (!current_user_can('manage_options')) {
            return;
        }
        $settings = self::get();
        ?>
        <div class="wrap">
            <h1>Ethnicity Detector</h1>
            <p>Use shortcode <code>[ethnicity_detector]</code> on any page.</p>
            <form method="post" action="options.php">
                <?php settings_fields('ed_settings_group'); ?>
                <table class="form-table" role="presentation">
                    <tr>
                        <th scope="row"><label for="ed_api_url">DeepFace API URL</label></th>
                        <td>
                            <input type="url" class="regular-text" id="ed_api_url" name="<?php echo esc_attr(self::OPTION_KEY); ?>[api_url]" value="<?php echo esc_attr($settings['api_url']); ?>" placeholder="https://your-api-host.example.com" />
                            <p class="description">Base URL of the FastAPI server from this GitHub repo (<code>api.py</code>). Example: <code>https://xxx.hf.space</code></p>
                        </td>
                    </tr>
                    <tr>
                        <th scope="row"><label for="ed_api_key">API key (optional)</label></th>
                        <td>
                            <input type="password" class="regular-text" id="ed_api_key" name="<?php echo esc_attr(self::OPTION_KEY); ?>[api_key]" value="<?php echo esc_attr($settings['api_key']); ?>" autocomplete="off" />
                            <p class="description">Must match the <code>API_KEY</code> env var on the Python API if set.</p>
                        </td>
                    </tr>
                    <tr>
                        <th scope="row"><label for="ed_title">Widget title</label></th>
                        <td><input type="text" class="regular-text" id="ed_title" name="<?php echo esc_attr(self::OPTION_KEY); ?>[title]" value="<?php echo esc_attr($settings['title']); ?>" /></td>
                    </tr>
                    <tr>
                        <th scope="row"><label for="ed_subtitle">Widget subtitle</label></th>
                        <td><input type="text" class="large-text" id="ed_subtitle" name="<?php echo esc_attr(self::OPTION_KEY); ?>[subtitle]" value="<?php echo esc_attr($settings['subtitle']); ?>" /></td>
                    </tr>
                </table>
                <?php submit_button(); ?>
            </form>
        </div>
        <?php
    }
}
