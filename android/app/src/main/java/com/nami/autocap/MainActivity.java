package com.nami.autocap;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Color;
import android.net.Uri;
import android.os.Bundle;
import android.provider.OpenableColumns;
import android.database.Cursor;
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

public final class MainActivity extends Activity {

    private static final int PICK_VIDEO_REQUEST = 147;

    private TextView selectedVideoText;
    private TextView statusText;
    private Button processButton;
    private Uri selectedVideoUri;

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

        TextView title = text(
            "NAMI AutoCap",
            28,
            Color.WHITE
        );
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
            "Việc nặng sẽ chạy trên worker bên ngoài. Điện thoại chỉ chọn video, gửi yêu cầu và nhận kết quả.",
            15,
            Color.rgb(184, 190, 204)
        );
        root.addView(workerInfo, spaced());

        Button chooseButton = button("CHỌN VIDEO");
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

        Spinner modeSpinner = new Spinner(this);
        String[] modes = new String[] {
            "Tạo phụ đề tiếng Việt",
            "Phụ đề + lồng tiếng Việt",
            "Chỉ xuất file SRT"
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
        processButton.setOnClickListener(v -> showWorkerStatus());
        root.addView(processButton, spaced());

        ProgressBar progressBar = new ProgressBar(
            this,
            null,
            android.R.attr.progressBarStyleHorizontal
        );
        progressBar.setMax(100);
        progressBar.setProgress(0);
        root.addView(progressBar, spaced());

        statusText = text(
            "Trạng thái: APK đang ở bản thử giao diện. Worker dịch thật sẽ được nối ở bước tiếp theo.",
            14,
            Color.rgb(184, 190, 204)
        );
        root.addView(statusText, spaced());

        TextView version = text(
            "Phiên bản 0.1.0 • V147A",
            12,
            Color.rgb(130, 136, 150)
        );
        version.setGravity(Gravity.CENTER_HORIZONTAL);
        root.addView(version, spaced());

        scrollView.addView(root);
        return scrollView;
    }

    private void chooseVideo() {
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType("video/*");
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
            requestCode != PICK_VIDEO_REQUEST ||
            resultCode != RESULT_OK ||
            data == null ||
            data.getData() == null
        ) {
            return;
        }

        selectedVideoUri = data.getData();

        try {
            getContentResolver().takePersistableUriPermission(
                selectedVideoUri,
                Intent.FLAG_GRANT_READ_URI_PERMISSION
            );
        } catch (SecurityException ignored) {
        }

        selectedVideoText.setText(
            "Đã chọn: " + getDisplayName(selectedVideoUri)
        );

        processButton.setEnabled(true);
        processButton.setAlpha(1.0f);

        statusText.setText(
            "Trạng thái: Video đã sẵn sàng để gửi lên worker."
        );
    }

    private void showWorkerStatus() {
        if (selectedVideoUri == null) {
            Toast.makeText(
                this,
                "Chưa chọn video.",
                Toast.LENGTH_SHORT
            ).show();
            return;
        }

        statusText.setText(
            "Trạng thái: APK đã nhận video. Worker dịch thật chưa được kết nối trong V147A."
        );

        Toast.makeText(
            this,
            "Bản V147A đã chọn được video thành công.",
            Toast.LENGTH_LONG
        ).show();
    }

    private String getDisplayName(Uri uri) {
        String result = "video đã chọn";

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
            com.nami.autocap.R.drawable.button_background
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
            value * getResources().getDisplayMetrics().density
        );
    }
}
