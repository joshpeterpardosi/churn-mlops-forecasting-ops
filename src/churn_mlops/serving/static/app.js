(() => {
  "use strict";

  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  // ---------- reveal-on-scroll ----------
  const revealEls = document.querySelectorAll(".reveal");
  if (reduceMotion || !("IntersectionObserver" in window)) {
    revealEls.forEach((el) => el.classList.add("is-in"));
  } else {
    const io = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add("is-in");
            io.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.15 }
    );
    revealEls.forEach((el) => io.observe(el));
  }

  // ---------- USD formatting (presentation-layer only) ----------
  const usd = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD" });

  // ---------- nav scroll shadow ----------
  const navEl = document.getElementById("nav");
  const onScroll = () => navEl.classList.toggle("is-scrolled", window.scrollY > 4);
  document.addEventListener("scroll", onScroll, { passive: true });
  onScroll();

  // ---------- count-up number tween (presentation only, final value always exact) ----------
  function animateNumber(el, from, to, { duration = 700, format } = {}) {
    el.textContent = format(from);
    if (reduceMotion) {
      el.textContent = format(to);
      return;
    }
    const start = performance.now();
    let settled = false;
    function settle() {
      if (settled) return;
      settled = true;
      el.textContent = format(to);
    }
    function tick(now) {
      const t = Math.min(1, (now - start) / duration);
      const eased = 1 - Math.pow(1 - t, 3);
      el.textContent = format(from + (to - from) * eased);
      if (t < 1) requestAnimationFrame(tick);
      else settle();
    }
    requestAnimationFrame(tick);
    // Fallback in case rAF never fires (backgrounded/inactive tab) — the
    // final value must always land, animation is a bonus, not a dependency.
    setTimeout(settle, duration + 200);
  }

  function flashLive(cardEl) {
    cardEl.classList.remove("is-live");
    void cardEl.offsetWidth;
    cardEl.classList.add("is-live");
    setTimeout(() => cardEl.classList.remove("is-live"), 900);
  }

  function pulseStatus(el) {
    el.classList.remove("is-pulsing");
    void el.offsetWidth;
    el.classList.add("is-pulsing");
  }

  // ---------- safe JSON -> highlighted markup ----------
  function escapeHtml(str) {
    return str.replace(/[&<>"']/g, (c) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    }[c]));
  }

  function renderJson(obj) {
    const json = JSON.stringify(obj, null, 2);
    const escaped = escapeHtml(json);
    return escaped.replace(
      /(&quot;.*?&quot;)(:?)|(\btrue\b|\bfalse\b|\bnull\b)|(-?\d+\.?\d*)/g,
      (match, str, colon, bool, num) => {
        if (str) {
          const cls = colon ? "tok-key" : "tok-str";
          return `<span class="${cls}">${str}</span>${colon}`;
        }
        if (bool) return `<span class="tok-bool">${bool}</span>`;
        if (num) return `<span class="tok-num">${num}</span>`;
        return match;
      }
    );
  }

  function setStatus(el, state, label) {
    el.dataset.state = state;
    el.textContent = label;
    if (state === "ok") pulseStatus(el);
  }

  function setButtonState(btn, state) {
    btn.dataset.state = state;
    btn.disabled = state === "loading";
  }

  function markFieldValidity(input) {
    input.classList.remove("is-valid");
    if (input.checkValidity() && input.value !== "") input.classList.add("is-valid");
  }

  // ---------- hero: real live example call ----------
  async function runHeroDemo() {
    const body = document.getElementById("hero-body");
    const status = document.getElementById("hero-status");
    const examplePayload = {
      tenure: 8,
      Contract: "Month-to-month",
      MonthlyCharges: 74.4,
      TotalCharges: 612.3,
      InternetService: "Fiber optic",
      PaymentMethod: "Electronic check",
      TechSupport: "No",
      current_mrr: 74.4,
      trailing_3mo_avg_mrr: 71.1,
    };
    try {
      const res = await fetch("/predict/churn", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(examplePayload),
      });
      const data = await res.json();
      if (res.ok) {
        setStatus(status, "ok", `${res.status} OK`);
        body.innerHTML = renderJson({ request: examplePayload, response: data });
        flashLive(document.getElementById("hero-card"));
      } else {
        setStatus(status, "error", String(res.status));
        body.innerHTML = renderJson({ request: examplePayload, error: data });
      }
    } catch (err) {
      setStatus(status, "error", "ERR");
      body.textContent = "Could not reach /predict/churn: " + err.message;
    }
  }

  // ---------- churn form ----------
  const churnForm = document.getElementById("churn-form");
  const churnErrorEl = document.getElementById("churn-error");
  const churnSubmit = document.getElementById("churn-submit");
  const churnStatus = document.getElementById("churn-status");
  const churnBody = document.getElementById("churn-body");
  const churnCard = document.getElementById("churn-card");
  const churnRisk = document.getElementById("churn-risk");

  churnForm.addEventListener("input", (e) => {
    if (e.target.matches("input, select")) markFieldValidity(e.target);
  });

  // ---------- auto-calc Total Charges from tenure x monthly, until user overrides it ----------
  const tenureInput = churnForm.elements["tenure"];
  const monthlyInput = churnForm.elements["MonthlyCharges"];
  const totalInput = churnForm.elements["TotalCharges"];
  let totalChargesDirty = false;

  function syncTotalCharges() {
    if (totalChargesDirty) return;
    const tenure = parseFloat(tenureInput.value);
    const monthly = parseFloat(monthlyInput.value);
    if (Number.isFinite(tenure) && Number.isFinite(monthly)) {
      totalInput.value = (tenure * monthly).toFixed(2);
      markFieldValidity(totalInput);
    }
  }

  tenureInput.addEventListener("input", syncTotalCharges);
  monthlyInput.addEventListener("input", syncTotalCharges);
  totalInput.addEventListener("input", () => {
    totalChargesDirty = true;
  });

  churnForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    churnErrorEl.hidden = true;
    if (!churnForm.checkValidity()) {
      churnForm.reportValidity();
      return;
    }
    churnRisk.hidden = true;

    const formData = new FormData(churnForm);
    const payload = {
      tenure: parseInt(formData.get("tenure"), 10),
      Contract: formData.get("Contract"),
      MonthlyCharges: parseFloat(formData.get("MonthlyCharges")),
      TotalCharges: parseFloat(formData.get("TotalCharges")),
      InternetService: formData.get("InternetService"),
      PaymentMethod: formData.get("PaymentMethod"),
      TechSupport: formData.get("TechSupport"),
      current_mrr: parseFloat(formData.get("current_mrr")),
      trailing_3mo_avg_mrr: parseFloat(formData.get("trailing_3mo_avg_mrr")),
    };

    setButtonState(churnSubmit, "loading");
    setStatus(churnStatus, "loading", "…");

    try {
      const res = await fetch("/predict/churn", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (res.ok) {
        setButtonState(churnSubmit, "success");
        setStatus(churnStatus, "ok", `${res.status} OK`);
        const pct = (data.churn_probability * 100).toFixed(1) + "%";
        churnBody.innerHTML = renderJson({ ...data, churn_probability_pct: pct });
        flashLive(churnCard);
        churnRisk.hidden = false;
        churnRisk.dataset.risk = data.churn_prediction ? "high" : "low";
        const prevPct = parseFloat(churnRisk.dataset.lastPct || "0");
        animateNumber(churnRisk, prevPct, data.churn_probability * 100, {
          format: (v) => v.toFixed(1) + "% churn risk",
        });
        churnRisk.dataset.lastPct = String(data.churn_probability * 100);
      } else {
        setButtonState(churnSubmit, "error");
        setStatus(churnStatus, "error", String(res.status));
        churnErrorEl.hidden = false;
        churnErrorEl.textContent = typeof data.detail === "string" ? data.detail : "Request failed.";
        churnBody.innerHTML = renderJson(data);
      }
    } catch (err) {
      setButtonState(churnSubmit, "error");
      setStatus(churnStatus, "error", "ERR");
      churnErrorEl.hidden = false;
      churnErrorEl.textContent = "Network error: " + err.message;
    } finally {
      setTimeout(() => setButtonState(churnSubmit, "idle"), 1600);
    }
  });

  // ---------- forecast form ----------
  const forecastForm = document.getElementById("forecast-form");
  const forecastErrorEl = document.getElementById("forecast-error");
  const forecastSubmit = document.getElementById("forecast-submit");
  const forecastStatus = document.getElementById("forecast-status");
  const forecastBody = document.getElementById("forecast-body");
  const forecastUsd = document.getElementById("forecast-usd");
  const forecastCard = document.getElementById("forecast-card");

  forecastForm.addEventListener("input", (e) => {
    if (e.target.matches("input, select")) markFieldValidity(e.target);
  });

  forecastForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    forecastErrorEl.hidden = true;
    if (!forecastForm.checkValidity()) {
      forecastForm.reportValidity();
      return;
    }

    const horizon = parseInt(new FormData(forecastForm).get("horizon"), 10);

    setButtonState(forecastSubmit, "loading");
    setStatus(forecastStatus, "loading", "…");
    forecastUsd.hidden = true;

    try {
      const res = await fetch(`/forecast/mrr?horizon=${horizon}`);
      const data = await res.json();
      if (res.ok) {
        setButtonState(forecastSubmit, "success");
        setStatus(forecastStatus, "ok", `${res.status} OK`);
        forecastBody.innerHTML = renderJson(data);
        flashLive(forecastCard);
        forecastUsd.hidden = false;
        const prevTotal = parseFloat(forecastUsd.dataset.lastTotal || "0");
        animateNumber(forecastUsd, prevTotal, data.forecast_total_mrr, {
          format: (v) => usd.format(v) + " total MRR",
        });
        forecastUsd.dataset.lastTotal = String(data.forecast_total_mrr);
      } else {
        setButtonState(forecastSubmit, "error");
        setStatus(forecastStatus, "error", String(res.status));
        forecastErrorEl.hidden = false;
        forecastErrorEl.textContent = typeof data.detail === "string" ? data.detail : "Request failed.";
        forecastBody.innerHTML = renderJson(data);
      }
    } catch (err) {
      setButtonState(forecastSubmit, "error");
      setStatus(forecastStatus, "error", "ERR");
      forecastErrorEl.hidden = false;
      forecastErrorEl.textContent = "Network error: " + err.message;
    } finally {
      setTimeout(() => setButtonState(forecastSubmit, "idle"), 1600);
    }
  });

  runHeroDemo();
})();
