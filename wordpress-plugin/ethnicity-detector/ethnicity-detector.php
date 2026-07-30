<?php
/**
 * Plugin Name: Ethnicity Detector
 * Description: User-friendly ethnicity & emotion detector powered by the same DeepFace API as the GitHub app.
 * Version: 1.0.0
 * Author: What Is My Ethnicity
 * Text Domain: ethnicity-detector
 */

if (!defined('ABSPATH')) {
    exit;
}

define('ED_PLUGIN_VERSION', '1.0.0');
define('ED_PLUGIN_FILE', __FILE__);
define('ED_PLUGIN_DIR', plugin_dir_path(__FILE__));
define('ED_PLUGIN_URL', plugin_dir_url(__FILE__));

require_once ED_PLUGIN_DIR . 'includes/class-ed-settings.php';
require_once ED_PLUGIN_DIR . 'includes/class-ed-rest.php';
require_once ED_PLUGIN_DIR . 'includes/class-ed-shortcode.php';

add_action('plugins_loaded', static function () {
    ED_Settings::init();
    ED_REST::init();
    ED_Shortcode::init();
});
