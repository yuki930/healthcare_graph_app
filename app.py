
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from datetime import timedelta
import pytz
import matplotlib.font_manager as fm

# 日本語フォントの設定
def setup_japanese_font():
    """日本語フォントを設定する"""
    # Streamlit Cloud用のフォントパス
    font_paths = [
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
    ]

    # フォントファイルが存在するか確認
    for font_path in font_paths:
        try:
            if fm.fontManager.addfont(font_path):
                plt.rcParams['font.family'] = 'Noto Sans CJK JP'
                break
        except:
            continue
    else:
        # フォールバック：システムにある日本語フォントを探す
        plt.rcParams['font.family'] = ['Noto Sans CJK JP', 'Noto Sans JP', 'Yu Gothic', 'Hiragino Sans', 'Meiryo', 'IPAexGothic', 'sans-serif']

    # マイナス記号の豆腐対策
    plt.rcParams['axes.unicode_minus'] = False

# フォント設定を実行
setup_japanese_font()


st.set_page_config(page_title="体組成グラフメーカー", layout="wide")

st.title("体組成グラフメーカー")
st.caption("朝・夜の2本線＋週平均＋開始日起点の目標ライン（6ヶ月/1年）を自動生成")

# セッション状態の初期化
if "settings" not in st.session_state:
    st.session_state.settings = {
        "target_weight": 50.0,
        "target_bf": 22.0,
        "target_skeletal": 30.0,
        "target_fatmass": 15.0,
        "draw_6m": True,
        "draw_12m": True,
        "ma_days": 7,
        "period_days": 60,
        "y_min_weight": 50.0,
        "y_min_bf": 15.0,
        "y_min_skeletal": 25.0,
        "fatmass_margin": 2.0
    }
if "settings_loaded" not in st.session_state:
    st.session_state.settings_loaded = False

with st.sidebar:
    st.header("⚙️ 設定")

    # 設定CSVの読み込み
    st.subheader("設定の読み込み/保存")
    config_uploaded = st.file_uploader("設定CSVを読み込む", type=["csv"], key="config_upload")

    # ファイル名が変更されたら再読み込みを許可
    if "last_config_name" not in st.session_state:
        st.session_state.last_config_name = None

    current_file_name = config_uploaded.name if config_uploaded is not None else None
    if current_file_name != st.session_state.last_config_name:
        st.session_state.settings_loaded = False
        st.session_state.last_config_name = current_file_name

    if config_uploaded is not None and not st.session_state.settings_loaded:
        try:
            config_df = pd.read_csv(config_uploaded)
            if "setting" in config_df.columns and "value" in config_df.columns:
                for _, row in config_df.iterrows():
                    setting_name = row["setting"]
                    if setting_name in st.session_state.settings:
                        value = row["value"]
                        # 真偽値の処理
                        if isinstance(st.session_state.settings[setting_name], bool):
                            st.session_state.settings[setting_name] = bool(value) if isinstance(value, bool) else str(value).lower() == "true"
                        # 整数の処理
                        elif setting_name in ["ma_days", "period_days"]:
                            st.session_state.settings[setting_name] = int(value)
                        # 浮動小数点の処理
                        else:
                            st.session_state.settings[setting_name] = float(value)
                    # 古いperiod設定からperiod_daysへ変換（互換性のため）
                    elif setting_name == "period" and "period_days" not in config_df["setting"].values:
                        period_str = str(row["value"])
                        if period_str != "全期間":
                            days = int(period_str.replace("直近","").replace("日",""))
                            st.session_state.settings["period_days"] = days
                # ウィジェットのセッション状態も更新
                st.session_state["input_target_weight"] = st.session_state.settings["target_weight"]
                st.session_state["input_target_bf"] = st.session_state.settings["target_bf"]
                st.session_state["input_target_skeletal"] = st.session_state.settings["target_skeletal"]
                st.session_state["input_target_fatmass"] = st.session_state.settings["target_fatmass"]
                st.session_state["input_draw_6m"] = st.session_state.settings["draw_6m"]
                st.session_state["input_draw_12m"] = st.session_state.settings["draw_12m"]
                st.session_state["input_ma_days"] = st.session_state.settings["ma_days"]
                if "period_days" in st.session_state.settings:
                    st.session_state["input_period_days"] = st.session_state.settings["period_days"]
                st.session_state["input_y_min_weight"] = st.session_state.settings["y_min_weight"]
                st.session_state["input_y_min_bf"] = st.session_state.settings["y_min_bf"]
                st.session_state["input_y_min_skeletal"] = st.session_state.settings["y_min_skeletal"]
                st.session_state["input_fatmass_margin"] = st.session_state.settings["fatmass_margin"]
                st.session_state.settings_loaded = True
                st.success("設定を読み込みました")
                st.rerun()
            else:
                st.error("CSVには「setting」と「value」列が必要です")
        except Exception as e:
            st.error(f"設定の読み込みに失敗しました: {e}")

    st.divider()

    # 各設定項目
    target_weight = st.number_input("目標体重 (kg)", value=st.session_state.settings["target_weight"], step=0.1, key="input_target_weight")
    target_bf = st.number_input("目標体脂肪率 (%)", value=st.session_state.settings["target_bf"], step=0.1, key="input_target_bf")
    target_skeletal = st.number_input("目標骨格筋率 (%)", value=st.session_state.settings["target_skeletal"], step=0.1, key="input_target_skeletal")
    target_fatmass = st.number_input("目標体脂肪量 (kg)", value=st.session_state.settings["target_fatmass"], step=0.1, key="input_target_fatmass")
    draw_6m = st.checkbox("6ヶ月到達ラインを表示", value=st.session_state.settings["draw_6m"], key="input_draw_6m")
    draw_12m = st.checkbox("1年到達ラインを表示", value=st.session_state.settings["draw_12m"], key="input_draw_12m")
    ma_days = st.number_input("週平均（日数）", min_value=3, max_value=30, value=st.session_state.settings["ma_days"], step=1, key="input_ma_days")

    st.subheader("表示期間設定")

    # データが読み込まれている場合のみ開始日を選択可能
    if "data_min_date" in st.session_state and "data_max_date" in st.session_state:
        # デフォルト値がmin/maxの範囲内に収まるように調整
        default_start_date = st.session_state.settings.get("start_date", st.session_state.data_min_date)
        if default_start_date < st.session_state.data_min_date:
            default_start_date = st.session_state.data_min_date
        elif default_start_date > st.session_state.data_max_date:
            default_start_date = st.session_state.data_max_date

        start_date_input = st.date_input(
            "表示開始日",
            value=default_start_date,
            min_value=st.session_state.data_min_date,
            max_value=st.session_state.data_max_date,
            key="input_start_date"
        )
        st.session_state.settings["start_date"] = start_date_input
    else:
        st.write("表示開始日: データ読み込み後に選択可能")

    period_days = st.number_input("表示日数", min_value=7, max_value=365, value=st.session_state.settings.get("period_days", 60), step=1, key="input_period_days")

    y_min_weight = st.number_input("体重グラフの下限 (kg)", value=st.session_state.settings["y_min_weight"], step=0.5, key="input_y_min_weight")
    y_min_bf = st.number_input("体脂肪率グラフの下限 (%)", value=st.session_state.settings["y_min_bf"], step=0.5, key="input_y_min_bf")
    y_min_skeletal = st.number_input("骨格筋率グラフの下限 (%)", value=st.session_state.settings["y_min_skeletal"], step=0.5, key="input_y_min_skeletal")
    fatmass_margin = st.number_input("体脂肪量グラフの余白 (kg)", value=st.session_state.settings["fatmass_margin"], step=0.5, key="input_fatmass_margin")

    # 設定を記憶
    st.session_state.settings.update({
        "target_weight": target_weight,
        "target_bf": target_bf,
        "target_skeletal": target_skeletal,
        "target_fatmass": target_fatmass,
        "draw_6m": draw_6m,
        "draw_12m": draw_12m,
        "ma_days": ma_days,
        "period_days": period_days,
        "y_min_weight": y_min_weight,
        "y_min_bf": y_min_bf,
        "y_min_skeletal": y_min_skeletal,
        "fatmass_margin": fatmass_margin
    })

    # 設定CSVのダウンロード
    st.divider()
    if st.button("現在の設定をCSVで保存"):
        # start_dateは除外（日付オブジェクトはCSV保存に適さない）
        settings_to_save = {k: v for k, v in st.session_state.settings.items() if k != "start_date"}
        settings_data = {
            "setting": list(settings_to_save.keys()),
            "value": list(settings_to_save.values())
        }
        settings_df = pd.DataFrame(settings_data)
        csv_buffer = io.StringIO()
        settings_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="設定CSVをダウンロード",
            data=csv_buffer.getvalue(),
            file_name="body_composition_settings.csv",
            mime="text/csv"
        )

st.markdown("### 1) CSVアップロード")
uploaded = st.file_uploader("CSVファイルを選択（列名は「測定日」「体重(kg)」「体脂肪(%)」「体脂肪量(kg)」「骨格筋(%)」等を想定）", type=["csv"])

def weekly_ma(series: pd.Series, window_days: int) -> pd.Series:
    daily = series.resample("D").mean().interpolate("time")
    return daily.rolling(window=window_days, min_periods=max(3, window_days//2)).mean()

def plot_metric(df: pd.DataFrame, col: str, ylabel: str, target_value: float,
                start_date, draw_6m: bool, draw_12m: bool, ma_days: int,
                y_min=None, y_max=None, title: str = "", x_range_days=None):
    fig, ax = plt.subplots(figsize=(9,5))

    # 背景色を設定
    ax.set_facecolor('#f8f9fa')
    fig.patch.set_facecolor('white')

    am = df[df["is_morning"]]
    pm = df[~df["is_morning"]]

    # ポイントを小さくし、色を改善
    ax.plot(am["datetime"], am[col], marker="o", markersize=4, linestyle="-",
            linewidth=1.5, color='#2E86AB', alpha=0.8, label="朝")
    ax.plot(pm["datetime"], pm[col], marker="o", markersize=4, linestyle="-",
            linewidth=1.5, color='#A23B72', alpha=0.8, label="夜")

    s = pd.Series(df[col].values, index=df["datetime"])
    ma = weekly_ma(s, ma_days)
    ax.plot(ma.index, ma.values, linewidth=3, color='#F18F01',
            alpha=0.9, label=f"{ma_days}日移動平均")

    ax.axhline(target_value, linestyle="--", linewidth=2, color="#6C757D",
               alpha=0.7, label=f"目標 {target_value}{ylabel[-1]}")

    start_val = df.iloc[0][col]
    if draw_6m:
        ax.plot([start_date, start_date + pd.DateOffset(months=6)], [start_val, target_value],
                linestyle=":", linewidth=2, color="#FF6B35", alpha=0.6, label="6ヶ月到達線")
    if draw_12m:
        ax.plot([start_date, start_date + pd.DateOffset(months=12)], [start_val, target_value],
                linestyle=":", linewidth=2, color="#004E89", alpha=0.6, label="1年到達線")

    ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold', pad=15)

    # Y軸の範囲を設定
    if (y_min is not None) or (y_max is not None):
        ymin = y_min if y_min is not None else min(df[col].min(), (ma.min() if not ma.isna().all() else df[col].min()))
        ymax = y_max if y_max is not None else max(df[col].max(), (ma.max() if not ma.isna().all() else df[col].max()))
        ax.set_ylim(ymin, ymax)

    # Y軸のメジャーグリッド（1キロごと）とマイナーグリッド（0.5キロごと）を設定
    from matplotlib.ticker import MultipleLocator
    ax.yaxis.set_major_locator(MultipleLocator(1.0))
    ax.yaxis.set_minor_locator(MultipleLocator(0.5))

    # グリッド線の設定
    ax.grid(True, which='major', linestyle='-', linewidth=0.8, color='#dee2e6', alpha=0.7)
    ax.grid(True, which='minor', linestyle='-', linewidth=0.4, color='#dee2e6', alpha=0.4)

    # 凡例のスタイル改善
    ax.legend(loc="best", frameon=True, fancybox=True, shadow=True,
              framealpha=0.95, fontsize=9)

    if x_range_days is not None:
        # 指定された日数範囲でX軸を固定
        # 開始日から指定日数分を表示
        xmin = df["datetime"].min()
        xmax = xmin + pd.Timedelta(days=x_range_days)
    else:
        # データ範囲に応じて余白を調整
        xmin = df["datetime"].min()
        data_days = (df["datetime"].max() - df["datetime"].min()).days
        margin_months = 1 if data_days <= 60 else 3
        xmax = df["datetime"].max() + pd.DateOffset(months=margin_months)
    ax.set_xlim(xmin, xmax)

    # X軸の日付を7日単位に設定
    from matplotlib.dates import DayLocator
    ax.xaxis.set_major_locator(DayLocator(interval=7))

    # 軸の境界線を目立たせる
    for spine in ax.spines.values():
        spine.set_edgecolor('#adb5bd')
        spine.set_linewidth(1.2)

    fig.autofmt_xdate(rotation=45)
    fig.tight_layout()
    return fig

if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
    except UnicodeDecodeError:
        uploaded.seek(0)
        df = pd.read_csv(uploaded, encoding="cp932")
    if "測定日" not in df.columns:
        st.error("CSVに「測定日」列が見つかりません。")
        st.stop()
    df["datetime"] = pd.to_datetime(df["測定日"])
    df = df.sort_values("datetime").reset_index(drop=True)
    df["is_morning"] = df["datetime"].dt.hour < 12
    if "体脂肪量(kg)" not in df.columns and {"体重(kg)","体脂肪(%)"}.issubset(df.columns):
        df["体脂肪量(kg)"] = df["体重(kg)"] * df["体脂肪(%)"] / 100.0

    # データの日付範囲をセッション状態に保存
    data_min_date = df["datetime"].min().date()
    data_max_date = df["datetime"].max().date()

    if st.session_state.get("data_min_date") != data_min_date or st.session_state.get("data_max_date") != data_max_date:
        st.session_state.data_min_date = data_min_date
        st.session_state.data_max_date = data_max_date
        st.rerun()

    # サイドバーで選択された開始日を取得
    start_date_input = st.session_state.settings.get("start_date", data_min_date)

    # データをフィルタ
    start_datetime = pd.Timestamp(start_date_input)
    end_datetime = start_datetime + pd.Timedelta(days=period_days)

    original_count = len(df)
    df = df[(df["datetime"] >= start_datetime) & (df["datetime"] < end_datetime)].copy()

    if df.empty:
        st.warning("選択した期間にデータがありません。")
        st.stop()

    start_date = start_datetime
    x_range_days = period_days
    # 体重
    if "体重(kg)" in df.columns:
        y_max_weight = max(df["体重(kg)"].max()+1, y_min_weight+1)
        title = f"体重推移（朝・夜別＋移動平均＋目標）\n期間: {df['datetime'].min().date()} 〜 {df['datetime'].max().date()}"
        fig = plot_metric(df, "体重(kg)", "体重 (kg)", target_weight, start_date, draw_6m, draw_12m, ma_days,
                          y_min=y_min_weight, y_max=y_max_weight,
                          title=title, x_range_days=x_range_days)
        st.pyplot(fig, clear_figure=True)
    # 体脂肪率
    if "体脂肪(%)" in df.columns:
        y_max_bf = max(df["体脂肪(%)"].max()+1, y_min_bf+1)
        fig = plot_metric(df, "体脂肪(%)", "体脂肪率 (%)", target_bf, start_date, draw_6m, draw_12m, ma_days,
                          y_min=y_min_bf, y_max=y_max_bf,
                          title="体脂肪率推移（朝・夜別＋移動平均＋目標）", x_range_days=x_range_days)
        st.pyplot(fig, clear_figure=True)
    # 骨格筋率
    if "骨格筋(%)" in df.columns:
        y_max_skeletal = max(df["骨格筋(%)"].max()+1, y_min_skeletal+1)
        fig = plot_metric(df, "骨格筋(%)", "骨格筋率 (%)", target_skeletal, start_date, draw_6m, draw_12m, ma_days,
                          y_min=y_min_skeletal, y_max=y_max_skeletal,
                          title="骨格筋率推移（朝・夜別＋移動平均＋目標）", x_range_days=x_range_days)
        st.pyplot(fig, clear_figure=True)
    # 体脂肪量
    if "体脂肪量(kg)" in df.columns:
        ymin = df["体脂肪量(kg)"].min() - fatmass_margin
        ymax = df["体脂肪量(kg)"].max() + fatmass_margin
        fig = plot_metric(df, "体脂肪量(kg)", "体脂肪量 (kg)", target_fatmass, start_date, draw_6m, draw_12m, ma_days,
                          y_min=ymin, y_max=ymax, title="体脂肪量推移（朝・夜別＋移動平均＋目標）", x_range_days=x_range_days)
        st.pyplot(fig, clear_figure=True)
else:
    st.info("左の設定を確認し、CSVをアップロードしてください。")
