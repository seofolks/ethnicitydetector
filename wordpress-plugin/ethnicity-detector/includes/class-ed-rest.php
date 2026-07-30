<?php

if (!defined('ABSPATH')) {
    exit;
}

class ED_REST
{
    public static function init(): void
    {
        add_action('rest_api_init', [self::class, 'register_routes']);
    }

    public static function register_routes(): void
    {
        register_rest_route('ethnicity-detector/v1', '/analyze', [
            'methods' => 'POST',
            'callback' => [self::class, 'analyze'],
            'permission_callback' => '__return_true',
        ]);
    }

    public static function analyze(WP_REST_Request $request)
    {
        $settings = ED_Settings::get();
        $api_url = untrailingslashit($settings['api_url'] ?? '');

        if ($api_url === '') {
            return new WP_Error(
                'ed_missing_api',
                'DeepFace API URL is not configured. Set it under Settings → Ethnicity Detector.',
                ['status' => 503]
            );
        }

        $files = $request->get_file_params();
        if (empty($files['image']) || empty($files['image']['tmp_name'])) {
            return new WP_Error('ed_missing_image', 'Please upload an image.', ['status' => 400]);
        }

        $file = $files['image'];
        $allowed = ['image/jpeg', 'image/png', 'image/webp', 'image/jpg'];
        $filetype = wp_check_filetype($file['name']);
        $mime = $file['type'] ?: ($filetype['type'] ?? '');

        if ($mime && !in_array($mime, $allowed, true)) {
            return new WP_Error('ed_bad_type', 'Supported formats: JPG, PNG, WEBP.', ['status' => 400]);
        }

        if (!empty($file['size']) && (int) $file['size'] > 8 * MB_IN_BYTES) {
            return new WP_Error('ed_too_large', 'Image must be under 8MB.', ['status' => 400]);
        }

        $endpoint = $api_url . '/analyze';
        $headers = [];
        if (!empty($settings['api_key'])) {
            $headers['X-API-Key'] = $settings['api_key'];
        }

        if (!function_exists('curl_init')) {
            return new WP_Error('ed_no_curl', 'PHP cURL is required.', ['status' => 500]);
        }

        $cfile = new CURLFile($file['tmp_name'], $mime ?: 'application/octet-stream', $file['name']);
        $ch = curl_init($endpoint);
        curl_setopt_array($ch, [
            CURLOPT_POST => true,
            CURLOPT_POSTFIELDS => ['file' => $cfile],
            CURLOPT_RETURNTRANSFER => true,
            CURLOPT_TIMEOUT => 180,
            CURLOPT_HTTPHEADER => array_map(
                static fn($k, $v) => $k . ': ' . $v,
                array_keys($headers),
                array_values($headers)
            ),
        ]);

        $body = curl_exec($ch);
        $errno = curl_errno($ch);
        $error = curl_error($ch);
        $status = (int) curl_getinfo($ch, CURLINFO_HTTP_CODE);
        curl_close($ch);

        if ($errno) {
            return new WP_Error('ed_curl', 'Could not reach DeepFace API: ' . $error, ['status' => 502]);
        }

        $decoded = json_decode((string) $body, true);
        if ($status >= 400) {
            $detail = is_array($decoded) ? ($decoded['detail'] ?? $body) : $body;
            if (is_array($detail)) {
                $detail = wp_json_encode($detail);
            }
            return new WP_Error('ed_api', 'Analysis failed: ' . $detail, ['status' => $status ?: 502]);
        }

        if (!is_array($decoded)) {
            return new WP_Error('ed_bad_response', 'Invalid API response.', ['status' => 502]);
        }

        return rest_ensure_response($decoded);
    }
}
