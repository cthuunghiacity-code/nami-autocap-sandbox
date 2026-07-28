package com.nami.autocap;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.Service;
import android.content.ContentValues;
import android.content.Intent;
import android.content.SharedPreferences;
import android.net.Uri;
import android.os.Build;
import android.os.Environment;
import android.os.IBinder;
import android.provider.MediaStore;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;

public final class BackgroundProcessingService extends Service {

    public static final String ACTION_STATUS =
        "com.nami.autocap.ACTION_STATUS";

    public static final String EXTRA_VIDEO_URI =
        "video_uri";

    public static final String EXTRA_VIDEO_NAME =
        "video_name";

    public static final String EXTRA_MODE =
        "mode";

    public static final String EXTRA_VOICE =
        "voice";

    public static final String EXTRA_STATE =
        "state";

    public static final String EXTRA_MESSAGE =
        "message";

    public static final String EXTRA_OUTPUT_URI =
        "output_uri";

    private static final String CHANNEL_ID =
        "nami_autocap_processing";

    private static final int NOTIFICATION_ID = 147;

    private static final String WORKER_URL =
        "https://cthuunghiacity-nc-ai-money-worker.hf.space"
            + "/autocap/process";

    private static final String PREFS =
        "nami_background_state";

    private volatile boolean processing;

    @Override
    public void onCreate() {
        super.onCreate();
        createNotificationChannel();
    }

    @Override
    public int onStartCommand(
        Intent intent,
        int flags,
        int startId
    ) {
        if (intent == null) {
            stopSelf();
            return START_NOT_STICKY;
        }

        if (processing) {
            return START_REDELIVER_INTENT;
        }

        String uriText =
            intent.getStringExtra(EXTRA_VIDEO_URI);

        String videoName =
            intent.getStringExtra(EXTRA_VIDEO_NAME);

        String mode =
            intent.getStringExtra(EXTRA_MODE);

        String voice =
            intent.getStringExtra(EXTRA_VOICE);

        if (
            uriText == null
                || uriText.trim().isEmpty()
        ) {
            stopSelf();
            return START_NOT_STICKY;
        }

        if (videoName == null || videoName.isEmpty()) {
            videoName = "video.mp4";
        }

        if (mode == null || mode.isEmpty()) {
            mode = "subtitle";
        }

        if (voice == null || voice.isEmpty()) {
            voice = "vi-VN-HoaiMyNeural";
        }

        final Uri videoUri = Uri.parse(uriText);
        final String finalVideoName = videoName;
        final String finalMode = mode;
        final String finalVoice = voice;

        processing = true;

        startForeground(
            NOTIFICATION_ID,
            buildNotification(
                "NAMI đang chuẩn bị xử lý video…",
                true
            )
        );

        saveState(
            true,
            "Đang chuẩn bị xử lý video…",
            ""
        );

        new Thread(() -> {
            try {
                sendStatus(
                    "uploading",
                    "Đang tải video lên worker…",
                    ""
                );

                updateNotification(
                    "Đang tải video lên worker…",
                    true
                );

                Uri output = uploadAndSave(
                    videoUri,
                    finalVideoName,
                    finalMode,
                    finalVoice
                );

                String outputText =
                    output == null
                        ? ""
                        : output.toString();

                saveState(
                    false,
                    "Hoàn tất: Video đã được lưu trong "
                        + "Movies/NAMI AutoCap.",
                    outputText
                );

                sendStatus(
                    "completed",
                    "Hoàn tất: Video đã được lưu trong "
                        + "Movies/NAMI AutoCap.",
                    outputText
                );

                updateNotification(
                    "Đã xử lý xong video",
                    false
                );

            } catch (Exception error) {
                String message =
                    error.getMessage() == null
                        ? error.getClass().getSimpleName()
                        : error.getMessage();

                saveState(
                    false,
                    "Lỗi xử lý: " + message,
                    ""
                );

                sendStatus(
                    "error",
                    "Lỗi xử lý: " + message,
                    ""
                );

                updateNotification(
                    "Xử lý video thất bại",
                    false
                );

            } finally {
                processing = false;

                stopForeground(false);
                stopSelf(startId);
            }
        }, "NamiAutoCapBackground").start();

        return START_REDELIVER_INTENT;
    }

    private Uri uploadAndSave(
        Uri selectedVideoUri,
        String selectedVideoName,
        String mode,
        String voice
    ) throws Exception {
        String boundary =
            "----NamiAutoCap"
                + System.currentTimeMillis();

        HttpURLConnection connection =
            (HttpURLConnection)
                new URL(WORKER_URL).openConnection();

        connection.setRequestMethod("POST");
        connection.setDoOutput(true);
        connection.setConnectTimeout(60_000);
        connection.setReadTimeout(3_600_000);
        connection.setChunkedStreamingMode(
            1024 * 1024
        );

        connection.setRequestProperty(
            "X-NAMI-Key",
            getString(R.string.nami_autocap_key)
        );

        connection.setRequestProperty(
            "Content-Type",
            "multipart/form-data; boundary="
                + boundary
        );

        try (
            OutputStream rawOutput =
                new BufferedOutputStream(
                    connection.getOutputStream()
                )
        ) {
            String safeName =
                selectedVideoName.replace(
                    "\"",
                    "_"
                );

            String fields =
                "--" + boundary + "\r\n"
                    + "Content-Disposition: form-data; "
                    + "name=\"mode\"\r\n\r\n"
                    + mode + "\r\n"
                    + "--" + boundary + "\r\n"
                    + "Content-Disposition: form-data; "
                    + "name=\"voice\"\r\n\r\n"
                    + voice + "\r\n";

            rawOutput.write(
                fields.getBytes(
                    StandardCharsets.UTF_8
                )
            );

            String header =
                "--" + boundary + "\r\n"
                    + "Content-Disposition: form-data; "
                    + "name=\"video\"; filename=\""
                    + safeName + "\"\r\n"
                    + "Content-Type: video/mp4"
                    + "\r\n\r\n";

            rawOutput.write(
                header.getBytes(
                    StandardCharsets.UTF_8
                )
            );

            try (
                InputStream input =
                    new BufferedInputStream(
                        getContentResolver()
                            .openInputStream(
                                selectedVideoUri
                            )
                    )
            ) {
                if (input == null) {
                    throw new Exception(
                        "Không đọc được video đã chọn."
                    );
                }

                byte[] buffer =
                    new byte[1024 * 1024];

                int count;

                while (
                    (count = input.read(buffer)) != -1
                ) {
                    rawOutput.write(
                        buffer,
                        0,
                        count
                    );
                }
            }

            rawOutput.write(
                (
                    "\r\n--"
                        + boundary
                        + "--\r\n"
                ).getBytes(
                    StandardCharsets.UTF_8
                )
            );

            rawOutput.flush();
        }

        int responseCode =
            connection.getResponseCode();

        if (
            responseCode
                != HttpURLConnection.HTTP_OK
        ) {
            InputStream errorStream =
                connection.getErrorStream();

            String details =
                errorStream == null
                    ? ""
                    : readSmallText(errorStream);

            connection.disconnect();

            throw new Exception(
                "Worker trả lỗi "
                    + responseCode
                    + (
                        details.isEmpty()
                            ? ""
                            : ": " + details
                    )
            );
        }

        sendStatus(
            "downloading",
            "Worker đã xử lý xong, "
                + "đang tải video về máy…",
            ""
        );

        updateNotification(
            "Đang tải video kết quả về máy…",
            true
        );

        Uri outputUri = createOutputUri();

        try (
            InputStream response =
                new BufferedInputStream(
                    connection.getInputStream()
                );
            OutputStream output =
                new BufferedOutputStream(
                    getContentResolver()
                        .openOutputStream(outputUri)
                )
        ) {
            if (output == null) {
                throw new Exception(
                    "Không tạo được file kết quả."
                );
            }

            byte[] buffer =
                new byte[1024 * 1024];

            int count;

            while (
                (count = response.read(buffer)) != -1
            ) {
                output.write(
                    buffer,
                    0,
                    count
                );
            }

            output.flush();
        } catch (Exception error) {
            try {
                getContentResolver().delete(
                    outputUri,
                    null,
                    null
                );
            } catch (Exception ignored) {
            }

            throw error;

        } finally {
            connection.disconnect();
        }

        if (Build.VERSION.SDK_INT >= 29) {
            ContentValues completed =
                new ContentValues();

            completed.put(
                MediaStore.Video.Media.IS_PENDING,
                0
            );

            getContentResolver().update(
                outputUri,
                completed,
                null,
                null
            );
        }

        return outputUri;
    }

    private Uri createOutputUri()
        throws Exception {
        String timestamp =
            new SimpleDateFormat(
                "yyyyMMdd_HHmmss",
                Locale.US
            ).format(new Date());

        ContentValues values =
            new ContentValues();

        values.put(
            MediaStore.Video.Media.DISPLAY_NAME,
            "NAMI_AutoCap_"
                + timestamp
                + ".mp4"
        );

        values.put(
            MediaStore.Video.Media.MIME_TYPE,
            "video/mp4"
        );

        if (Build.VERSION.SDK_INT >= 29) {
            values.put(
                MediaStore.Video.Media.RELATIVE_PATH,
                Environment.DIRECTORY_MOVIES
                    + "/NAMI AutoCap"
            );

            values.put(
                MediaStore.Video.Media.IS_PENDING,
                1
            );
        }

        Uri result =
            getContentResolver().insert(
                MediaStore.Video.Media
                    .EXTERNAL_CONTENT_URI,
                values
            );

        if (result == null) {
            throw new Exception(
                "Không tạo được video trong thư viện."
            );
        }

        return result;
    }

    private String readSmallText(
        InputStream input
    ) throws Exception {
        try (
            InputStream source = input;
            ByteArrayOutputStream output =
                new ByteArrayOutputStream()
        ) {
            byte[] buffer = new byte[4096];
            int count;
            int total = 0;

            while (
                (count = source.read(buffer)) != -1
                    && total < 20_000
            ) {
                int allowed =
                    Math.min(
                        count,
                        20_000 - total
                    );

                output.write(
                    buffer,
                    0,
                    allowed
                );

                total += allowed;
            }

            return output.toString(
                StandardCharsets.UTF_8.name()
            ).trim();
        }
    }

    private void createNotificationChannel() {
        if (Build.VERSION.SDK_INT < 26) {
            return;
        }

        NotificationChannel channel =
            new NotificationChannel(
                CHANNEL_ID,
                "NAMI xử lý video",
                NotificationManager.IMPORTANCE_LOW
            );

        channel.setDescription(
            "Thông báo khi NAMI đang xử lý "
                + "video trong nền"
        );

        NotificationManager manager =
            getSystemService(
                NotificationManager.class
            );

        if (manager != null) {
            manager.createNotificationChannel(
                channel
            );
        }
    }

    private Notification buildNotification(
        String message,
        boolean ongoing
    ) {
        Intent openIntent =
            new Intent(
                this,
                MainActivity.class
            );

        openIntent.setFlags(
            Intent.FLAG_ACTIVITY_SINGLE_TOP
                | Intent.FLAG_ACTIVITY_CLEAR_TOP
        );

        PendingIntent pendingIntent =
            PendingIntent.getActivity(
                this,
                147,
                openIntent,
                PendingIntent.FLAG_UPDATE_CURRENT
                    | PendingIntent.FLAG_IMMUTABLE
            );

        Notification.Builder builder =
            Build.VERSION.SDK_INT >= 26
                ? new Notification.Builder(
                    this,
                    CHANNEL_ID
                )
                : new Notification.Builder(this);

        builder
            .setContentTitle("NAMI AutoCap")
            .setContentText(message)
            .setSmallIcon(
                android.R.drawable
                    .stat_sys_upload
            )
            .setContentIntent(pendingIntent)
            .setOngoing(ongoing)
            .setOnlyAlertOnce(true);

        if (!ongoing) {
            builder.setAutoCancel(true);
        }

        return builder.build();
    }

    private void updateNotification(
        String message,
        boolean ongoing
    ) {
        NotificationManager manager =
            (NotificationManager)
                getSystemService(
                    NOTIFICATION_SERVICE
                );

        if (manager != null) {
            manager.notify(
                NOTIFICATION_ID,
                buildNotification(
                    message,
                    ongoing
                )
            );
        }
    }

    private void sendStatus(
        String state,
        String message,
        String outputUri
    ) {
        Intent update =
            new Intent(ACTION_STATUS);

        update.setPackage(getPackageName());
        update.putExtra(EXTRA_STATE, state);
        update.putExtra(EXTRA_MESSAGE, message);
        update.putExtra(
            EXTRA_OUTPUT_URI,
            outputUri
        );

        sendBroadcast(update);
    }

    private void saveState(
        boolean running,
        String message,
        String outputUri
    ) {
        SharedPreferences preferences =
            getSharedPreferences(
                PREFS,
                MODE_PRIVATE
            );

        preferences
            .edit()
            .putBoolean("running", running)
            .putString("message", message)
            .putString(
                "output_uri",
                outputUri
            )
            .apply();
    }

    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }
}
