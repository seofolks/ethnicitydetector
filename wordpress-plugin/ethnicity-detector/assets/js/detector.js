(function () {
  function qs(root, sel) {
    return root.querySelector(sel);
  }

  function setStatus(el, message, isError) {
    if (!message) {
      el.hidden = true;
      el.textContent = "";
      el.classList.remove("is-error");
      return;
    }
    el.hidden = false;
    el.textContent = message;
    el.classList.toggle("is-error", !!isError);
  }

  function sortedScores(scores) {
    return Object.entries(scores || {}).sort(function (a, b) {
      return b[1] - a[1];
    });
  }

  function renderScores(title, scores) {
    var rows = sortedScores(scores)
      .map(function (entry) {
        var name = entry[0];
        var value = Number(entry[1]) || 0;
        var pct = Math.max(0, Math.min(100, value));
        return (
          '<div class="ed-score">' +
          '<span class="ed-score__name">' +
          name +
          "</span>" +
          '<div class="ed-score__bar"><div class="ed-score__fill" style="width:' +
          pct +
          '%"></div></div>' +
          '<span class="ed-score__pct">' +
          pct.toFixed(1) +
          "%</span>" +
          "</div>"
        );
      })
      .join("");

    return (
      '<div class="ed-card"><h3>' +
      title +
      "</h3>" +
      rows +
      "</div>"
    );
  }

  function renderResults(container, data) {
    container.hidden = false;
    container.innerHTML =
      '<div class="ed-card ed-dominant">' +
      '<div class="ed-dominant__item"><span class="ed-dominant__label">Dominant ethnicity</span><span class="ed-dominant__value">' +
      (data.dominant_ethnicity || "n/a") +
      "</span></div>" +
      '<div class="ed-dominant__item"><span class="ed-dominant__label">Dominant emotion</span><span class="ed-dominant__value">' +
      (data.dominant_emotion || "n/a") +
      "</span></div>" +
      "</div>" +
      renderScores("Ethnicity scores", data.ethnicity_scores) +
      renderScores("Emotion scores", data.emotion_scores);
  }

  function init(root) {
    var cfg = window.ED_DETECTOR || {};
    var fileInput = qs(root, "[data-ed-file]");
    var drop = qs(root, "[data-ed-drop]");
    var preview = qs(root, "[data-ed-preview]");
    var previewImg = qs(root, "[data-ed-preview-img]");
    var analyzeBtn = qs(root, "[data-ed-analyze]");
    var statusEl = qs(root, "[data-ed-status]");
    var resultsEl = qs(root, "[data-ed-results]");
    var video = qs(root, "[data-ed-video]");
    var canvas = qs(root, "[data-ed-canvas]");
    var startCam = qs(root, "[data-ed-start-cam]");
    var captureBtn = qs(root, "[data-ed-capture]");
    var selectedFile = null;
    var stream = null;

    function setFile(file) {
      if (!file || !file.type || file.type.indexOf("image/") !== 0) {
        setStatus(statusEl, "Please choose a JPG, PNG, or WEBP image.", true);
        return;
      }
      selectedFile = file;
      preview.hidden = false;
      previewImg.src = URL.createObjectURL(file);
      analyzeBtn.disabled = !cfg.hasApi;
      resultsEl.hidden = true;
      setStatus(statusEl, "");
    }

    root.querySelectorAll("[data-ed-mode]").forEach(function (btn) {
      btn.addEventListener("click", function () {
        var mode = btn.getAttribute("data-ed-mode");
        root.querySelectorAll("[data-ed-mode]").forEach(function (b) {
          var active = b === btn;
          b.classList.toggle("is-active", active);
          b.setAttribute("aria-selected", active ? "true" : "false");
        });
        root.querySelectorAll("[data-ed-panel]").forEach(function (panel) {
          panel.hidden = panel.getAttribute("data-ed-panel") !== mode;
        });
        if (mode !== "webcam" && stream) {
          stream.getTracks().forEach(function (t) {
            t.stop();
          });
          stream = null;
          captureBtn.disabled = true;
        }
      });
    });

    drop.addEventListener("click", function () {
      fileInput.click();
    });
    drop.addEventListener("dragover", function (e) {
      e.preventDefault();
      drop.classList.add("is-dragover");
    });
    drop.addEventListener("dragleave", function () {
      drop.classList.remove("is-dragover");
    });
    drop.addEventListener("drop", function (e) {
      e.preventDefault();
      drop.classList.remove("is-dragover");
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        setFile(e.dataTransfer.files[0]);
      }
    });
    fileInput.addEventListener("change", function () {
      if (fileInput.files && fileInput.files[0]) {
        setFile(fileInput.files[0]);
      }
    });

    startCam.addEventListener("click", async function () {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "user" },
          audio: false,
        });
        video.srcObject = stream;
        captureBtn.disabled = false;
        setStatus(statusEl, "");
      } catch (err) {
        setStatus(statusEl, "Could not access webcam. Check browser permissions.", true);
      }
    });

    captureBtn.addEventListener("click", function () {
      if (!stream) return;
      canvas.width = video.videoWidth || 640;
      canvas.height = video.videoHeight || 480;
      var ctx = canvas.getContext("2d");
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      canvas.toBlob(function (blob) {
        if (!blob) return;
        setFile(new File([blob], "webcam.jpg", { type: "image/jpeg" }));
      }, "image/jpeg", 0.92);
    });

    analyzeBtn.addEventListener("click", async function () {
      if (!selectedFile) return;
      if (!cfg.hasApi) {
        setStatus(statusEl, "API URL is not configured in plugin settings.", true);
        return;
      }

      analyzeBtn.disabled = true;
      setStatus(statusEl, "Analyzing with DeepFace… first run can take a minute.");
      resultsEl.hidden = true;

      var form = new FormData();
      form.append("image", selectedFile, selectedFile.name || "photo.jpg");

      try {
        var res = await fetch(cfg.restUrl, {
          method: "POST",
          headers: { "X-WP-Nonce": cfg.nonce },
          body: form,
        });
        var payload = await res.json().catch(function () {
          return null;
        });
        if (!res.ok) {
          var msg =
            (payload && (payload.message || payload.code)) ||
            "Analysis failed (" + res.status + ").";
          throw new Error(msg);
        }
        setStatus(statusEl, "Analysis complete.");
        renderResults(resultsEl, payload);
      } catch (err) {
        setStatus(statusEl, err.message || "Analysis failed.", true);
      } finally {
        analyzeBtn.disabled = false;
      }
    });
  }

  document.addEventListener("DOMContentLoaded", function () {
    document.querySelectorAll("[data-ed-root]").forEach(init);
  });
})();
