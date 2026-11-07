import io
import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="مساعد الأسهم", page_icon="📈", layout="centered")
st.markdown("<style>html{direction:rtl}</style>", unsafe_allow_html=True)

TOL = 0.01  # أقصى فرق مسموح بين المصادر (1%)


# ---------- المصادر ----------
def src_yahoo(t):
    df = yf.download(t, period="3y", auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()


def src_stooq(t):
    sym = t.lower() if "." in t else t.lower() + ".us"
    r = requests.get(f"https://stooq.com/q/d/l/?s={sym}&i=d", timeout=15)
    df = pd.read_csv(io.StringIO(r.text), parse_dates=["Date"], index_col="Date")
    if df.empty or "Close" not in df:
        raise ValueError("no data")
    return df.tail(750)


def src_twelve(t):
    key = st.secrets.get("TWELVE_KEY", "")
    if not key:
        raise ValueError("no key")
    r = requests.get("https://api.twelvedata.com/time_series",
                     params={"symbol": t, "interval": "1day", "outputsize": 750, "apikey": key},
                     timeout=15).json()
    df = pd.DataFrame(r["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime").sort_index().astype(float)
    df.columns = [c.capitalize() for c in df.columns]
    return df


SOURCES = {"Yahoo Finance": src_yahoo, "Stooq": src_stooq, "Twelve Data": src_twelve}


def fetch_all(t):
    data, errors = {}, {}
    for name, fn in SOURCES.items():
        try:
            d = fn(t)
            if len(d) > 100:
                data[name] = d
            else:
                errors[name] = "بيانات قليلة"
        except Exception as e:
            errors[name] = "مفيش بيانات/مش متاح"
    return data, errors


def verify(data):
    """يقارن آخر 10 أيام إقفال بين المصادر. يرجّع المصادر اللي متطابقة مع Yahoo (أو أول مصدر)."""
    names = list(data)
    base = names[0]
    agree = [base]
    details = {}
    for n in names[1:]:
        a = data[base]["Close"].tail(30)
        b = data[n]["Close"]
        idx = a.index.intersection(b.index)[-10:]
        if len(idx) < 5:
            details[n] = None
            continue
        diff = float(((a[idx] - b[idx]).abs() / a[idx]).max())
        details[n] = diff
        if diff <= TOL:
            agree.append(n)
    return base, agree, details


# ---------- المؤشرات ----------
def indicators(df):
    d = df.copy()
    c = d["Close"]
    d["ema20"] = c.ewm(span=20, adjust=False).mean()
    d["ema50"] = c.ewm(span=50, adjust=False).mean()
    delta = c.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    d["rsi"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    d["macdh"] = macd - macd.ewm(span=9, adjust=False).mean()
    tr = pd.concat([d["High"] - d["Low"],
                    (d["High"] - c.shift()).abs(),
                    (d["Low"] - c.shift()).abs()], axis=1).max(axis=1)
    d["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    d["entry"] = ((d["ema20"] > d["ema50"]) & (c > d["ema20"]) &
                  (d["macdh"] > 0) & d["rsi"].between(45, 70))
    return d.dropna()


def backtest(d, sl_mult, rr, max_days=60):
    trades, i = [], 0
    rows = d.reset_index()
    n = len(rows)
    while i < n - 1:
        r = rows.iloc[i]
        if not r["entry"]:
            i += 1
            continue
        entry = r["Close"]
        stop = entry - sl_mult * r["atr"]
        target = entry + rr * sl_mult * r["atr"]
        out, days = "timeout", max_days
        for k in range(1, max_days + 1):
            if i + k >= n:
                out, days = None, None
                break
            lo, hi = rows.iloc[i + k]["Low"], rows.iloc[i + k]["High"]
            if lo <= stop:
                out, days = "loss", k
                break
            if hi >= target:
                out, days = "win", k
                break
        if out is None:
            break
        ret = {"win": rr * sl_mult * r["atr"], "loss": -sl_mult * r["atr"]}.get(
            out, rows.iloc[min(i + max_days, n - 1)]["Close"] - entry) / entry
        trades.append({"out": out, "days": days, "ret": ret})
        i += days + 1
    return pd.DataFrame(trades)


# ---------- الواجهة ----------
st.title("📈 مساعد الأسهم")
st.caption("إشارات مبنية على قواعد + backtest. مش توصية مالية ومش مضمونة.")

ticker = st.text_input("رمز السهم", "AAPL",
                       help="أمريكي: AAPL | مصري: COMI.CA | سعودي: 2222.SR").strip().upper()
c1, c2 = st.columns(2)
sl_mult = c1.slider("وقف الخسارة (× ATR)", 1.0, 3.0, 1.5, 0.5)
rr = c2.slider("نسبة العائد للمخاطرة", 1.0, 4.0, 2.0, 0.5)

if st.button("حلّل", type="primary", use_container_width=True) and ticker:
    with st.spinner("بجيب البيانات من كل المصادر وبقارن..."):
        data, errors = fetch_all(ticker)

    if not data:
        st.error("ملقيتش بيانات للرمز ده في أي مصدر. اتأكد من الرمز.")
        st.stop()

    base, agree, details = verify(data)

    st.subheader("1) التحقق من المصادر")
    for n in SOURCES:
        if n in errors:
            st.write(f"⚪ {n}: {errors[n]}")
        elif n == base:
            st.write(f"🟢 {n}: المصدر الأساسي — آخر إقفال {data[n]['Close'].iloc[-1]:.2f}")
        elif details.get(n) is None:
            st.write(f"⚪ {n}: مفيش تواريخ مشتركة كفاية للمقارنة")
        elif n in agree:
            st.write(f"🟢 {n}: متطابق (أقصى فرق {details[n]*100:.2f}%)")
        else:
            st.write(f"🔴 {n}: مختلف (فرق {details[n]*100:.2f}%)")

    if len(agree) < 2:
        st.warning("⛔ مفيش مصدرين على الأقل متطابقين، فمش هديك إشارة. "
                   "ده مقصود عشان متتبنيش قرار على بيانات مش متأكدة. "
                   "ملحوظة: بعض الأسواق (زي البورصة المصرية) مغطاة من مصدر واحد في المصادر المجانية؛ "
                   "ممكن تضيف مفتاح Twelve Data في secrets.")
        st.stop()

    d = indicators(data[base])
    last = d.iloc[-1]
    bt = backtest(d, sl_mult, rr)

    st.subheader("2) الإشارة")
    price = float(last["Close"])
    stop = price - sl_mult * float(last["atr"])
    target = price + rr * sl_mult * float(last["atr"])

    if bt.empty or len(bt) < 8:
        st.info("الـ backtest فيه صفقات قليلة (أقل من 8)، النتيجة مش موثوقة.")

    win_rate = (bt["out"] == "win").mean() if len(bt) else 0
    expectancy = bt["ret"].mean() if len(bt) else 0

    if bool(last["entry"]) and expectancy > 0:
        st.success("✅ إشارة دخول")
        m1, m2, m3 = st.columns(3)
        m1.metric("الدخول", f"{price:.2f}")
        m2.metric("وقف الخسارة", f"{stop:.2f}", f"-{(price-stop)/price*100:.1f}%")
        m3.metric("الهدف", f"{target:.2f}", f"+{(target-price)/price*100:.1f}%")
        wins = bt[bt["out"] == "win"]
        if len(wins):
            st.write(f"⏱️ المدة المتوقعة للوصول للهدف: حوالي **{int(wins['days'].median())} يوم** "
                     "(وسيط الصفقات الناجحة تاريخياً)")
    elif bool(last["entry"]):
        st.warning("🟡 شروط الدخول متحققة لكن القاعدة دي أداؤها التاريخي على السهم ده سلبي. متدخلش.")
    else:
        trend = "صاعد" if last["ema20"] > last["ema50"] else "هابط/جانبي"
        st.warning(f"⏳ استنى. شروط الدخول مش متحققة دلوقتي (الاتجاه {trend}، RSI={last['rsi']:.0f}).")

    st.subheader("3) نتيجة الـ backtest (3 سنين)")
    if len(bt):
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("عدد الصفقات", len(bt))
        b2.metric("نسبة النجاح", f"{win_rate*100:.0f}%")
        b3.metric("متوسط الصفقة", f"{expectancy*100:.2f}%")
        b4.metric("أسوأ صفقة", f"{bt['ret'].min()*100:.1f}%")
        st.caption(f"متوسط مدة الصفقة {bt['days'].mean():.0f} يوم. "
                   "النتايج الماضية مش ضمان للمستقبل، والـ backtest مفيهوش عمولات ولا انزلاق سعري.")
    else:
        st.write("مفيش صفقات في الفترة دي.")

    st.line_chart(d[["Close", "ema20", "ema50"]].tail(250))
