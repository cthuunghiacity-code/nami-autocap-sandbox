package com.nami.autocap;

import android.app.Activity;
import android.content.ContentValues;
import android.content.Intent;
import android.database.Cursor;
import android.graphics.Color;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.provider.OpenableColumns;
import android.view.Gravity;
import android.view.View;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.ScrollView;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;

public final class MainActivity extends Activity {

    private static final int PICK_VIDEO_REQUEST = 147;

    private static final String WORKER_URL =
        "https://cthuunghiacity-nc-ai-money-worker.hf.space"
        + "/autocap/process";

    private TextView selectedVideoText;
    private TextView statusText;
    private Button chooseButton;
    private Button processButton;
    private ProgressBar progressBar;
    private Spinner modeSpinner;

    private Uri selectedVideoUri;
    private String selectedVideoName = "video.mp4";
    private volatile boolean processing = false;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(createContent());
    }

    private View createContent() {
        int padding = dp(20);

        ScrollView scrollView = new ScrollView(this);
        scrollView.setBackgroundColor(Color.rgb(16, 18, 24));

        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(padding, padding, padding, padding);

        TextView title = text("NAMI AutoCap", 28, Color.WHITE);
        title.setGravity(Gravity.CENTER_HORIZONTAL);
        root.addView(title, matchWrap());

        TextView subtitle = text(
            "Dịch video tiếng Trung sang tiếng Việt",
            16,
            Color.rgb(184, 190, 204)
        );
        subtitle.setGravity(Gravity.CENTER_HORIZONTAL);
        root.addView(subtitle, spaced());

        TextView workerInfo = text(
            "Video được gửi lên worker bên ngoài. "
                + "Điện thoại không chạy mô hình nặng.",
            15,
            Color.rgb(184, 190, 204)
        );
        root.addView(workerInfo, spaced());

        chooseButton = button("CHỌN VIDEO");
        chooseButton.setOnClickListener(v -> chooseVideo());
        root.addView(chooseButton, spaced());

        selectedVideoText = text(
            "Chưa chọn video",
            15,
            Color.WHITE
        );
        root.addView(selectedVideoText, spaced());

        TextView modeLabel = text(
            "Chế độ xử lý",
            16,
            Color.WHITE
        );
        root.addView(modeLabel, spaced());

        modeSpinner = new Spinner(this);

        String[] modes = new String[] {
            "Tạo video có phụ đề tiếng Việt",
            "Phụ đề + lồng tiếng Việt nữ",
            "Phụ đề + lồng tiếng Việt nam"
        };

        ArrayAdapter<String> adapter = new ArrayAdapter<>(
            this,
            android.R.layout.simple_spinner_dropdown_item,
            modes
        );

        modeSpinner.setAdapter(adapter);
        root.addView(modeSpinner, matchWrap());

        processButton = button("GỬI XỬ LÝ");
        processButton.setEnabled(false);
        processButton.setAlpha(0.45f);
        processButton.setOnClickListener(v -> startProcessing());
        root.addView(processButton, spaced());

        progressBar = new ProgressBar(
            this,
            null,
            android.R.attr.progressBarStyleHorizontal
        );

        progressBar.setMax(100);
        progressBar.setProgress(0);
        root.addView(progressBar, spaced());

        statusText = text(
            "Trạng thái: Chờ chọn video.",
            14,
            Color.rgb(184, 190, 204)
        );
        root.addView(statusText, spaced());

        TextView version = text(
            "Phiên bản 0.3.0 • V147D",
            12,
            Color.rgb(130, 136, 150)
        );
        version.setGravity(Gravity.CENTER_HORIZONTAL);
        root.addView(version, spaced());

        scrollView.addView(root);
        return scrollView;
    }

    private void chooseVideo() {
        if (processing) {
            return;
        }

        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType("video/*");
        intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
        intent.addFlags(Intent.FLAG_GRANT_PERSISTABLE_URI_PERMISSION);

        startActivityForResult(intent, PICK_VIDEO_REQUEST);
    }

    @Override
    protected void onActivityResult(
        int requestCode,
        int resultCode,
        Intent data
    ) {
        super.onActivityResult(requestCode, resultCode, data);

        if (
            requestCode != PICK_VIDEO_REQUEST
                || resultCode != RESULT_OK
                || data == null
                || data.getData() == null
        ) {
            return;
        }

        selectedVideoUri = data.getData();
        selectedVideoName = getDisplayName(selectedVideoUri);

        try {
            getContentResolver().takePersistableUriPermission(
                selectedVideoUri,
                Intent.FLAG_GRANT_READ_URI_PERMISSION
            );
        } catch (Exception ignored) {
        }

        selectedVideoText.setText(
            "Đã chọn: " + selectedVideoName
        );

        processButton.setEnabled(true);
        processButton.setAlpha(1.0f);
        progressBar.setProgress(0);

        statusText.setText(
            "Trạng thái: Video sẵn sàng gửi lên worker."
        );
    }

    private void startProcessing() {
        if (selectedVideoUri == null || processing) {
            return;
        }

        processing = true;
        setControlsEnabled(false);
        progressBar.setIndeterminate(true);

        statusText.setText(
            "Trạng thái: Đang tải video lên worker…"
        );

        new Thread(() -> {
            try {
                Uri result = uploadAndSave();

                runOnUiThread(() -> {
                    processing = false;
                    progressBar.setIndeterminate(false);
                    progressBar.setProgress(100);
                    setControlsEnabled(true);

                    statusText.setText(
                        "Hoàn tất: Video đã được lưu tại "
                            + "Movies/NAMI AutoCap/."
                    );

                    Toast.makeText(
                        this,
                        "Đã lưu video trong Movies/NAMI AutoCap",
                        Toast.LENGTH_LONG
                    ).show();

                    openResult(result);
                });
            } catch (Exception error) {
                final String message =
                    error.getMessage() == null
                        ? error.getClass().getSimpleName()
                        : error.getMessage();

                runOnUiThread(() -> {
                    processing = false;
                    progressBar.setIndeterminate(false);
                    progressBar.setProgress(0);
                    setControlsEnabled(true);

                    statusText.setText(
                        "Lỗi xử lý: " + message
                    );

                    Toast.makeText(
                        this,
                        "Không xử lý được video.",
                        Toast.LENGTH_LONG
                    ).show();
                });
            }
        }).start();
    }

    private Uri uploadAndSave() throws Exception {
        String boundary =
            "----NamiAutoCap" + System.currentTimeMillis();

        HttpURLConnection connection =
            (HttpURLConnection) new URL(WORKER_URL)
                .openConnection();

        connection.setRequestMethod("POST");
        connection.setDoOutput(true);
        connection.setConnectTimeout(60_000);
        connection.setReadTimeout(1_800_000);
        connection.setChunkedStreamingMode(1024 * 1024);

        connection.setRequestProperty(
            "X-NAMI-Key",
            getString(R.string.nami_autocap_key)
        );

        connection.setRequestProperty(
            "Content-Type",
            "multipart/form-data; boundary=" + boundary
        );

        try (
            OutputStream rawOutput =
                new BufferedOutputStream(
                    connection.getOutputStream()
                )
        ) {
            String safeName =
                selectedVideoName.replace("\"", "_");

            int selectedMode =
                modeSpinner.getSelectedItemPosition();

            String mode =
                selectedMode == 0
                    ? "subtitle"
                    : "dub";

            String voice =
                selectedMode == 2
                    ? "vi-VN-NamMinhNeural"
                    : "vi-VN-HoaiMyNeural";

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
                    + "Content-Type: video/mp4\r\n\r\n";

            rawOutput.write(
                header.getBytes(StandardCharsets.UTF_8)
            );

            try (
                InputStream input =
                    new BufferedInputStream(
                        getContentResolver()
                            .openInputStream(selectedVideoUri)
                    )
            ) {
                if (input == null) {
                    throw new Exception(
                        "Không đọc được video đã chọn."
                    );
                }

                byte[] buffer = new byte[1024 * 1024];
                int count;

                while ((count = input.read(buffer)) != -1) {
                    rawOutput.write(buffer, 0, count);
                }
            }

            rawOutput.write(
                ("\r\n--" + boundary + "--\r\n")
                    .getBytes(StandardCharsets.UTF_8)
            );

            rawOutput.flush();
        }

        int responseCode = connection.getResponseCode();

        if (responseCode != HttpURLConnection.HTTP_OK) {
            InputStream errorStream =
                connection.getErrorStream();

            String details =
                errorStream == null
                    ? ""
                    : readSmallText(errorStream);

            throw new Exception(
                "Worker trả lỗi "
                    + responseCode
                    + (details.isEmpty()
                        ? ""
                        : ": " + details)
            );
        }

        runOnUiThread(() ->
            statusText.setText(
                "Trạng thái: Worker xử lý xong, "
                    + "đang lưu video về máy…"
            )
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
                    "Không mở được file kết quả."
                );
            }

            byte[] buffer = new byte[1024 * 1024];
            int count;

            while ((count = response.read(buffer)) != -1) {
                output.write(buffer, 0, count);
            }

            output.flush();
        }

        finalizeOutputUri(outputUri);
        connection.disconnect();

        return outputUri;
    }

    private Uri createOutputUri() throws Exception {
        String filename =
            "NAMI_AutoCap_vietsub_"
                + System.currentTimeMillis()
                + ".mp4";

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            ContentValues values = new ContentValues();
            values.put(
                MediaStore.Video.Media.DISPLAY_NAME,
                filename
            );
            values.put(
                MediaStore.Video.Media.MIME_TYPE,
                "video/mp4"
            );
            values.put(
                MediaStore.Video.Media.RELATIVE_PATH,
                Environment.DIRECTORY_MOVIES
                    + "/NAMI AutoCap"
            );
            values.put(
                MediaStore.Video.Media.IS_PENDING,
                1
            );

            Uri uri = getContentResolver().insert(
                MediaStore.Video.Media.EXTERNAL_CONTENT_URI,
                values
            );

            if (uri == null) {
                throw new Exception(
                    "Không tạo được file trong Download."
                );
            }

            return uri;
        }

        File directory = getExternalFilesDir(
            Environment.DIRECTORY_MOVIES
        );

        if (directory == null) {
            throw new Exception(
                "Không tìm được thư mục lưu video."
            );
        }

        File file = new File(directory, filename);
        return Uri.fromFile(file);
    }

    private void finalizeOutputUri(Uri uri) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            ContentValues values = new ContentValues();
            values.put(
                MediaStore.Video.Media.IS_PENDING,
                0
            );

            getContentResolver().update(
                uri,
                values,
                null,
                null
            );
        }
    }

    private void openResult(Uri uri) {
        try {
            Intent intent = new Intent(Intent.ACTION_VIEW);
            intent.setDataAndType(uri, "video/mp4");
            intent.addFlags(
                Intent.FLAG_GRANT_READ_URI_PERMISSION
            );
            startActivity(intent);
        } catch (Exception ignored) {
        }
    }

    private String readSmallText(
        InputStream input
    ) throws Exception {
        byte[] buffer = new byte[4096];
        int count = input.read(buffer);

        if (count <= 0) {
            return "";
        }

        return new String(
            buffer,
            0,
            count,
            StandardCharsets.UTF_8
        ).trim();
    }

    private void setControlsEnabled(boolean enabled) {
        chooseButton.setEnabled(enabled);
        processButton.setEnabled(
            enabled && selectedVideoUri != null
        );
        modeSpinner.setEnabled(enabled);

        chooseButton.setAlpha(enabled ? 1.0f : 0.45f);
        processButton.setAlpha(
            enabled && selectedVideoUri != null
                ? 1.0f
                : 0.45f
        );
    }

    private String getDisplayName(Uri uri) {
        String result = "video.mp4";

        Cursor cursor = getContentResolver().query(
            uri,
            null,
            null,
            null,
            null
        );

        if (cursor == null) {
            return result;
        }

        try {
            int index = cursor.getColumnIndex(
                OpenableColumns.DISPLAY_NAME
            );

            if (cursor.moveToFirst() && index >= 0) {
                String value = cursor.getString(index);

                if (value != null && !value.isEmpty()) {
                    result = value;
                }
            }
        } finally {
            cursor.close();
        }

        return result;
    }

    private TextView text(
        String value,
        int size,
        int color
    ) {
        TextView view = new TextView(this);
        view.setText(value);
        view.setTextSize(size);
        view.setTextColor(color);
        view.setLineSpacing(0f, 1.15f);
        return view;
    }

    private Button button(String value) {
        Button button = new Button(this);
        button.setText(value);
        button.setTextColor(Color.rgb(16, 18, 24));
        button.setTextSize(15);
        button.setAllCaps(false);
        button.setBackgroundResource(
            R.drawable.button_background
        );
        return button;
    }

    private LinearLayout.LayoutParams matchWrap() {
        return new LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT,
            LinearLayout.LayoutParams.WRAP_CONTENT
        );
    }

    private LinearLayout.LayoutParams spaced() {
        LinearLayout.LayoutParams params = matchWrap();
        params.topMargin = dp(18);
        return params;
    }

    private int dp(int value) {
        return Math.round(
            value
                * getResources()
                    .getDisplayMetrics()
                    .density
        );
    }
}
