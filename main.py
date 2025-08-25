#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Haze-aware Face & Eye Detection (Full, PySide6)
- Fixed-size QLabel(안 커지게)
- Preprocess: BGR/Gray/CLAHE/Blur(gauss, median, bilateral)/Sharpen/Laplacian/Sobel/Canny/Unsharp/CustomKernel
- Blend: Original vs Processed
- MediaPipe FaceMesh EAR + Drowsy alarm (threshold/hold frames), overlay + red banner
- Menu: File/Tools/Help, Custom Kernel Editor Dialog(미리보기/정규화/적용)
- Non-blocking: QThread worker + safe shutdown

Author: ChatGPT for 오빠 (2025-08-26)
"""
import sys, os, time, math, traceback
import cv2
import numpy as np

from PySide6 import QtCore, QtGui, QtWidgets

# ----------------------------- Utils ---------------------------------
def bgr_to_qimage(bgr: np.ndarray) -> QtGui.QImage:
    if bgr is None or bgr.size == 0:
        return QtGui.QImage()
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    return QtGui.QImage(rgb.data, w, h, ch*w, QtGui.QImage.Format.Format_RGB888).copy()

def letterbox_fit(img: np.ndarray, target_w: int, target_h: int, pad_color=(24,24,24)) -> np.ndarray:
    if img is None or img.size == 0 or target_w <=0 or target_h <=0:
        return np.full((max(1,target_h), max(1,target_w), 3), pad_color, np.uint8)
    h, w = img.shape[:2]
    s = min(target_w/w, target_h/h)
    nw, nh = int(w*s), int(h*s)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if s<1 else cv2.INTER_LINEAR)
    out = np.full((target_h, target_w, 3), pad_color, np.uint8)
    x0 = (target_w - nw)//2; y0 = (target_h - nh)//2
    out[y0:y0+nh, x0:x0+nw] = resized
    return out

def draw_banner(img, text="DROWSY!", color=(0,0,255)):
    h, w = img.shape[:2]
    bar_h = max(28, h//18)
    cv2.rectangle(img, (0,0), (w, bar_h), color, -1)
    cv2.putText(img, text, (12, int(bar_h*0.72)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2, cv2.LINE_AA)

def put_kv(img, x, y, s, v, col=(255,255,255)):
    cv2.putText(img, f"{s}: {v}", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)

# ----------------------------- EAR / FaceMesh -------------------------
FACE_EYE_L = [33, 160, 158, 133, 153, 144]   # outer points around left eye
FACE_EYE_R = [362, 385, 387, 263, 373, 380]
# EAR 계산용 6포인트: [p1,p2,p3,p4,p5,p6]
def eye_aspect_ratio(pts):  # pts: 6x2
    # EAR = (||p2-p6|| + ||p3-p5||) / (2*||p1-p4||)
    p1,p2,p3,p4,p5,p6 = [pts[i] for i in range(6)]
    def dist(a,b): return np.linalg.norm(a-b)
    ear = (dist(p2,p6) + dist(p3,p5)) / (2.0*dist(p1,p4) + 1e-6)
    return float(ear)

# ----------------------------- ImageView (fixed) ----------------------
class ImageView(QtWidgets.QLabel):
    def __init__(self, fixed_w=640, fixed_h=360):
        super().__init__()
        self._fixed_w, self._fixed_h = int(fixed_w), int(fixed_h)
        self.setMinimumSize(self._fixed_w, self._fixed_h)
        self.setMaximumSize(self._fixed_w, self._fixed_h)  # ✅ 안 커지게
        self.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setStyleSheet("QLabel { background:#111; color:#ddd; border:1px solid #333; }")
        self._last = None

    def setImage(self, bgr: np.ndarray):
        if bgr is None: return
        view = letterbox_fit(bgr, self._fixed_w, self._fixed_h)
        qimg = bgr_to_qimage(view)
        self._last = qimg
        self.setPixmap(QtGui.QPixmap.fromImage(qimg))

# ----------------------------- Params / State -------------------------
class Params(QtCore.QObject):
    changed = QtCore.Signal()
    def __init__(self):
        super().__init__()
        # Color / Enhancement
        self.color_mode = "BGR"      # BGR, Gray, CLAHE
        self.clahe_clip = 2.0
        self.clahe_tile = 8

        # Blur
        self.blur_mode = "Off"       # Off, Gaussian, Median, Bilateral
        self.gauss_ksize = 5
        self.gauss_sigma = 1.0
        self.median_ksize = 5
        self.bilateral_d = 7
        self.bilateral_sigma = 50

        # Edge / Sharpen
        self.edge_mode = "Off"       # Off, Laplacian, Sobel, Canny, Unsharp
        self.edge_intensity = 1.0    # 공용 강도
        self.canny_auto = True
        self.canny_alpha = 0.66      # 자동 임계 보정
        self.unsharp_amount = 0.6
        self.unsharp_radius = 1.5

        # Custom Kernel
        self.custom_kernel = None    # np.ndarray or None
        self.custom_norm = True

        # Blend
        self.blend = 100             # 0..100

        # Drowsy
        self.ear_thresh = 0.22       # 0.05..0.40
        self.ear_digits = 3
        self.drowsy_hold = 15        # frames
        self.overlay_eye = True
        self.overlay_box = True
        self.overlay_label = True

# ----------------------------- Worker (QThread target) ----------------
class Worker(QtCore.QObject):
    frameReady = QtCore.Signal(np.ndarray, np.ndarray, dict)   # orig, proc, stats
    logReady   = QtCore.Signal(str)

    @QtCore.Slot()
    def start(self):
        self._timer.start()

    @QtCore.Slot()
    def stop(self):
        try: self._timer.stop()
        except: pass
        # mediapipe 자원 반납
        if getattr(self, "_mp_face", None) is not None:
            try:
                self._mp_face.close()
            except: pass
            self._mp_face = None

    @QtCore.Slot(np.ndarray)
    def enqueue(self, frame):
        # 마지막 프레임 저장(드롭-최신)
        self._latest = frame

    def __init__(self, params: Params):
        super().__init__()
        self.params = params
        self._timer = QtCore.QTimer(self); self._timer.setInterval(10)
        self._timer.timeout.connect(self._on_tick)
        self._latest = None

        # mediapipe 초기화(지연 생성)
        self._mp_face = None
        self._drowsy_count = 0
        self._is_alarm = False

    # ------------------- Processing Pipeline -------------------
    def _apply_color(self, img, p: Params):
        if p.color_mode == "Gray":
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        elif p.color_mode == "CLAHE":
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            v = hsv[:, :, 2]
            clip = max(0.1, float(p.clahe_clip))
            tile = max(2, int(p.clahe_tile))
            clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
            hsv[:, :, 2] = clahe.apply(v)
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return img

    def _apply_blur(self, img, p: Params):
        if p.blur_mode == "Gaussian":
            k = int(p.gauss_ksize) | 1
            sigma = float(p.gauss_sigma)
            return cv2.GaussianBlur(img, (k, k), sigma)
        elif p.blur_mode == "Median":
            k = max(3, int(p.median_ksize) | 1)
            return cv2.medianBlur(img, k)
        elif p.blur_mode == "Bilateral":
            d = max(3, int(p.bilateral_d))
            s = max(1.0, float(p.bilateral_sigma))
            return cv2.bilateralFilter(img, d, s, s)
        return img

    def _apply_edge_sharp(self, img, p: Params):
        mode = p.edge_mode
        if mode == "Off":
            return img
        if mode == "Laplacian":
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            lap = cv2.Laplacian(g, cv2.CV_16S, ksize=3)
            lap = cv2.convertScaleAbs(lap, alpha=p.edge_intensity)
            return cv2.cvtColor(lap, cv2.COLOR_GRAY2BGR)
        if mode == "Sobel":
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            sx = cv2.Sobel(g, cv2.CV_16S, 1, 0, ksize=3)
            sy = cv2.Sobel(g, cv2.CV_16S, 0, 1, ksize=3)
            mag = cv2.convertScaleAbs(cv2.magnitude(sx.astype(np.float32), sy.astype(np.float32)))
            mag = cv2.convertScaleAbs(mag, alpha=p.edge_intensity)
            return cv2.cvtColor(mag, cv2.COLOR_GRAY2BGR)
        if mode == "Canny":
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if p.canny_auto:
                # median 기반 자동 임계
                med = np.median(g)
                lower = int(max(0, (1.0 - p.canny_alpha) * med))
                upper = int(min(255, (1.0 + p.canny_alpha) * med))
            else:
                lower = int(max(0, 50 * p.edge_intensity))
                upper = int(min(255, 150 * p.edge_intensity))
            edge = cv2.Canny(g, lower, upper)
            return cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)
        if mode == "Unsharp":
            blur = cv2.GaussianBlur(img, (0,0), p.unsharp_radius)
            return cv2.addWeighted(img, 1.0 + p.unsharp_amount, blur, -p.unsharp_amount, 0)
        return img

    def _apply_custom_kernel(self, img, p: Params):
        if p.custom_kernel is None:
            return img
        k = p.custom_kernel.astype(np.float32)
        if p.custom_norm:
            s = k.sum()
            if abs(s) > 1e-9:
                k = k / s
        return cv2.filter2D(img, -1, k)

    # ------------------- MediaPipe / EAR -----------------------
    def _ensure_facemesh(self):
        if self._mp_face is None:
            import mediapipe as mp
            self._mp_face = mp.solutions.face_mesh.FaceMesh(
                refine_landmarks=True, max_num_faces=1, static_image_mode=False
            )
            self._mp_drawing = mp.solutions.drawing_utils

    def _calc_ear(self, frame_bgr, p: Params, draw=True):
        self._ensure_facemesh()
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self._mp_face.process(rgb)
        ear_val = None
        rect = None
        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0]
            pts = np.array([(int(l.x*w), int(l.y*h)) for l in lm.landmark], dtype=np.int32)

            # Bounding box
            x0,y0 = pts[:,0].min(), pts[:,1].min()
            x1,y1 = pts[:,0].max(), pts[:,1].max()
            rect = (x0,y0,x1-x0,y1-y0)

            left = pts[FACE_EYE_L]
            right= pts[FACE_EYE_R]
            ear_l = eye_aspect_ratio(left.astype(np.float32))
            ear_r = eye_aspect_ratio(right.astype(np.float32))
            ear_val = (ear_l + ear_r) / 2.0

            if draw and p.overlay_eye:
                cv2.polylines(frame_bgr, [left], True, (50,200,240), 2, cv2.LINE_AA)
                cv2.polylines(frame_bgr, [right], True, (50,200,240), 2, cv2.LINE_AA)
            if draw and p.overlay_box and rect is not None:
                x,y,w0,h0 = rect
                cv2.rectangle(frame_bgr, (x,y), (x+w0, y+h0), (80,220,100), 2)
            if draw and p.overlay_label and ear_val is not None:
                put_kv(frame_bgr, 10, 30, "EAR", f"{ear_val:.{p.ear_digits}f}")
        return ear_val, rect

    # ------------------- Main tick -----------------------------
    def _on_tick(self):
        frame = self._latest
        if frame is None: return
        p = self.params
        orig = frame.copy()

        # 1) Preprocess
        x = self._apply_color(orig, p)
        x = self._apply_blur(x, p)
        x = self._apply_edge_sharp(x, p)
        x = self._apply_custom_kernel(x, p)

        # 2) Blend
        alpha = float(p.blend)/100.0
        x = cv2.addWeighted(x, alpha, orig, 1.0-alpha, 0)

        # 3) EAR & Drowsy
        ear, rect = self._calc_ear(x, p, draw=True)
        stats = {"ear": "-" if ear is None else f"{ear:.{p.ear_digits}f}"}

        if ear is not None:
            if ear < p.ear_thresh:
                self._drowsy_count += 1
            else:
                if self._is_alarm:
                    self.logReady.emit("ALARM OFF (recovered)")
                self._is_alarm = False
                self._drowsy_count = 0

            if self._drowsy_count >= int(p.drowsy_hold):
                if not self._is_alarm:
                    self._is_alarm = True
                    self.logReady.emit("ALARM ON (EAR below threshold)")
        else:
            # 얼굴이 사라지면 카운트 리셋
            if self._is_alarm:
                self.logReady.emit("ALARM OFF (no face)")
            self._is_alarm = False
            self._drowsy_count = 0

        if self._is_alarm:
            draw_banner(x, "DROWSY!", (0,0,255))

        # 4) Emit
        self.frameReady.emit(orig, x, stats)

# ----------------------------- Custom Kernel Dialog -------------------
class KernelDialog(QtWidgets.QDialog):
    kernelApplied = QtCore.Signal(np.ndarray, bool)  # kernel, normalize

    def __init__(self, get_preview_frame_cb):
        super().__init__()
        self.setWindowTitle("Custom Kernel Editor")
        self.resize(520, 420)
        self.get_preview = get_preview_frame_cb

        v = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()
        self.edit = QtWidgets.QPlainTextEdit()
        self.edit.setPlaceholderText("예) 0 -1 0\n-1 5 -1\n0 -1 0")
        self.chkNorm = QtWidgets.QCheckBox("Normalize (sum=1)")
        self.chkNorm.setChecked(True)
        form.addRow("Kernel", self.edit)
        form.addRow("", self.chkNorm)
        v.addLayout(form)

        # preview
        self.preview = ImageView(320, 180)
        v.addWidget(self.preview)

        h = QtWidgets.QHBoxLayout()
        self.btnPrev = QtWidgets.QPushButton("Preview")
        self.btnApply= QtWidgets.QPushButton("Apply")
        self.btnClose= QtWidgets.QPushButton("Close")
        h.addWidget(self.btnPrev); h.addWidget(self.btnApply); h.addStretch(1); h.addWidget(self.btnClose)
        v.addLayout(h)

        self.btnPrev.clicked.connect(self.on_preview)
        self.btnApply.clicked.connect(self.on_apply)
        self.btnClose.clicked.connect(self.accept)

    def parse_kernel(self):
        text = self.edit.toPlainText().strip()
        if not text:
            raise ValueError("빈 커널")
        rows = []
        for line in text.splitlines():
            line = line.strip().replace(",", " ")
            if not line: continue
            vals = [float(x) for x in line.split()]
            rows.append(vals)
        # 직사각 허용(odd 권장)
        maxc = max(len(r) for r in rows)
        for i,r in enumerate(rows):
            if len(r) < maxc:
                rows[i] = r + [0.0]*(maxc-len(r))
        k = np.array(rows, dtype=np.float32)
        if k.size == 0:
            raise ValueError("유효하지 않은 커널")
        return k

    @QtCore.Slot()
    def on_preview(self):
        try:
            k = self.parse_kernel()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Parse Error", str(e)); return
        img = self.get_preview()
        if img is None:
            QtWidgets.QMessageBox.information(self, "Preview", "프리뷰 이미지가 아직 없어요."); return
        kn = k.copy()
        if self.chkNorm.isChecked():
            s = kn.sum()
            if abs(s) > 1e-9: kn /= s
        y = cv2.filter2D(img, -1, kn)
        self.preview.setImage(y)

    @QtCore.Slot()
    def on_apply(self):
        try:
            k = self.parse_kernel()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Parse Error", str(e)); return
        self.kernelApplied.emit(k, self.chkNorm.isChecked())
        self.accept()

# ----------------------------- Control Panel --------------------------
class ControlPanel(QtWidgets.QWidget):
    paramsChanged = QtCore.Signal()

    def __init__(self, p: Params):
        super().__init__()
        self.p = p
        grid = QtWidgets.QGridLayout(self)
        r = 0

        # Source
        self.btnWebcam = QtWidgets.QPushButton("Webcam ▶")
        self.btnVideo  = QtWidgets.QPushButton("Open Video…")
        self.btnImage  = QtWidgets.QPushButton("Open Image…")
        self.btnStop   = QtWidgets.QPushButton("Stop ■")
        grid.addWidget(self.btnWebcam, r,0); grid.addWidget(self.btnVideo, r,1)
        grid.addWidget(self.btnImage, r,2);  grid.addWidget(self.btnStop, r,3); r+=1

        # Color
        grid.addWidget(QtWidgets.QLabel("Color Mode"), r,0)
        self.comboColor = QtWidgets.QComboBox(); self.comboColor.addItems(["BGR","Gray","CLAHE"])
        grid.addWidget(self.comboColor, r,1)
        self.spinClip = QtWidgets.QDoubleSpinBox(); self.spinClip.setRange(0.1,10.0); self.spinClip.setValue(p.clahe_clip); self.spinClip.setSingleStep(0.1)
        self.spinTile = QtWidgets.QSpinBox(); self.spinTile.setRange(2,32); self.spinTile.setValue(p.clahe_tile)
        grid.addWidget(QtWidgets.QLabel("CLAHE clip"), r,2); grid.addWidget(self.spinClip, r,3); r+=1
        grid.addWidget(QtWidgets.QLabel("CLAHE tile"), r,2); grid.addWidget(self.spinTile, r,3); r+=1

        # Blur
        grid.addWidget(QtWidgets.QLabel("Blur"), r,0)
        self.comboBlur = QtWidgets.QComboBox(); self.comboBlur.addItems(["Off","Gaussian","Median","Bilateral"])
        grid.addWidget(self.comboBlur, r,1)
        self.spinGk = QtWidgets.QSpinBox(); self.spinGk.setRange(1,31); self.spinGk.setSingleStep(2); self.spinGk.setValue(p.gauss_ksize)
        self.spinGs = QtWidgets.QDoubleSpinBox(); self.spinGs.setRange(0.1,10.0); self.spinGs.setValue(p.gauss_sigma)
        grid.addWidget(QtWidgets.QLabel("G ksize"), r,2); grid.addWidget(self.spinGk, r,3); r+=1
        grid.addWidget(QtWidgets.QLabel("G sigma"), r,2); grid.addWidget(self.spinGs, r,3); r+=1

        self.spinMedian = QtWidgets.QSpinBox(); self.spinMedian.setRange(3,31); self.spinMedian.setSingleStep(2); self.spinMedian.setValue(p.median_ksize)
        grid.addWidget(QtWidgets.QLabel("Median ksize"), r,2); grid.addWidget(self.spinMedian, r,3); r+=1
        self.spinBd = QtWidgets.QSpinBox(); self.spinBd.setRange(3,21); self.spinBd.setValue(p.bilateral_d)
        self.spinBs = QtWidgets.QDoubleSpinBox(); self.spinBs.setRange(1.0,200.0); self.spinBs.setValue(p.bilateral_sigma)
        grid.addWidget(QtWidgets.QLabel("Bilateral d"), r,2); grid.addWidget(self.spinBd, r,3); r+=1
        grid.addWidget(QtWidgets.QLabel("Bilateral σ"), r,2); grid.addWidget(self.spinBs, r,3); r+=1

        # Edge/Sharpen
        grid.addWidget(QtWidgets.QLabel("Edge/Sharpen"), r,0)
        self.comboEdge = QtWidgets.QComboBox(); self.comboEdge.addItems(["Off","Laplacian","Sobel","Canny","Unsharp"])
        grid.addWidget(self.comboEdge, r,1)
        self.spinInt = QtWidgets.QDoubleSpinBox(); self.spinInt.setRange(0.1,5.0); self.spinInt.setValue(p.edge_intensity); self.spinInt.setSingleStep(0.1)
        self.chkCannyAuto = QtWidgets.QCheckBox("Canny auto"); self.chkCannyAuto.setChecked(True)
        self.spinCAlpha = QtWidgets.QDoubleSpinBox(); self.spinCAlpha.setRange(0.1,1.0); self.spinCAlpha.setSingleStep(0.05); self.spinCAlpha.setValue(p.canny_alpha)
        grid.addWidget(QtWidgets.QLabel("Intensity"), r,2); grid.addWidget(self.spinInt, r,3); r+=1
        grid.addWidget(self.chkCannyAuto, r,2); grid.addWidget(self.spinCAlpha, r,3); r+=1
        self.spinUSAmt = QtWidgets.QDoubleSpinBox(); self.spinUSAmt.setRange(0.0,3.0); self.spinUSAmt.setSingleStep(0.05); self.spinUSAmt.setValue(p.unsharp_amount)
        self.spinUSRad = QtWidgets.QDoubleSpinBox(); self.spinUSRad.setRange(0.1,5.0); self.spinUSRad.setSingleStep(0.1); self.spinUSRad.setValue(p.unsharp_radius)
        grid.addWidget(QtWidgets.QLabel("Unsharp amt"), r,2); grid.addWidget(self.spinUSAmt, r,3); r+=1
        grid.addWidget(QtWidgets.QLabel("Unsharp rad"), r,2); grid.addWidget(self.spinUSRad, r,3); r+=1

        # Blend
        self.sliderBlend = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.sliderBlend.setRange(0,100); self.sliderBlend.setValue(p.blend)
        grid.addWidget(QtWidgets.QLabel("Blend (proc %)"), r,0); grid.addWidget(self.sliderBlend, r,1,1,3); r+=1

        # Drowsy
        grid.addWidget(QtWidgets.QLabel("EAR thresh"), r,0)
        self.spinEAR = QtWidgets.QDoubleSpinBox(); self.spinEAR.setRange(0.05,0.40); self.spinEAR.setSingleStep(0.01); self.spinEAR.setValue(p.ear_thresh)
        grid.addWidget(self.spinEAR, r,1)
        grid.addWidget(QtWidgets.QLabel("Hold frames"), r,2)
        self.spinHold = QtWidgets.QSpinBox(); self.spinHold.setRange(1,120); self.spinHold.setValue(p.drowsy_hold)
        grid.addWidget(self.spinHold, r,3); r+=1

        self.chkEye = QtWidgets.QCheckBox("Draw eye"); self.chkEye.setChecked(p.overlay_eye)
        self.chkBox = QtWidgets.QCheckBox("Draw face box"); self.chkBox.setChecked(p.overlay_box)
        self.chkLbl = QtWidgets.QCheckBox("Draw EAR label"); self.chkLbl.setChecked(p.overlay_label)
        grid.addWidget(self.chkEye, r,1); grid.addWidget(self.chkBox, r,2); grid.addWidget(self.chkLbl, r,3); r+=1

        grid.setColumnStretch(4,1)

        # bindings
        self.comboColor.currentTextChanged.connect(self.emit_params)
        for w in [self.spinClip, self.spinTile,
                  self.comboBlur, self.spinGk, self.spinGs, self.spinMedian, self.spinBd, self.spinBs,
                  self.comboEdge, self.spinInt, self.chkCannyAuto, self.spinCAlpha, self.spinUSAmt, self.spinUSRad,
                  self.sliderBlend, self.spinEAR, self.spinHold, self.chkEye, self.chkBox, self.chkLbl]:
            if isinstance(w, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox, QtWidgets.QSlider)):
                w.valueChanged.connect(self.emit_params)
            elif isinstance(w, QtWidgets.QCheckBox):
                w.toggled.connect(self.emit_params)
            elif isinstance(w, QtWidgets.QComboBox):
                w.currentTextChanged.connect(self.emit_params)

    def emit_params(self):
        p = self.p
        p.color_mode = self.comboColor.currentText()
        p.clahe_clip = float(self.spinClip.value()); p.clahe_tile = int(self.spinTile.value())

        p.blur_mode = self.comboBlur.currentText()
        p.gauss_ksize = int(self.spinGk.value()); p.gauss_sigma = float(self.spinGs.value())
        p.median_ksize = int(self.spinMedian.value())
        p.bilateral_d = int(self.spinBd.value()); p.bilateral_sigma = float(self.spinBs.value())

        p.edge_mode = self.comboEdge.currentText()
        p.edge_intensity = float(self.spinInt.value())
        p.canny_auto = bool(self.chkCannyAuto.isChecked()); p.canny_alpha = float(self.spinCAlpha.value())
        p.unsharp_amount = float(self.spinUSAmt.value()); p.unsharp_radius = float(self.spinUSRad.value())

        p.blend = int(self.sliderBlend.value())

        p.ear_thresh = float(self.spinEAR.value())
        p.drowsy_hold = int(self.spinHold.value())
        p.overlay_eye = self.chkEye.isChecked()
        p.overlay_box = self.chkBox.isChecked()
        p.overlay_label = self.chkLbl.isChecked()

        p.changed.emit()
        self.paramsChanged.emit()

# ----------------------------- Video Source ---------------------------
class VideoSource(QtCore.QObject):
    frameCaptured = QtCore.Signal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.cap = None
        self.timer = QtCore.QTimer(self); self.timer.setInterval(30)
        self.timer.timeout.connect(self._on_timeout)

    def open_webcam(self, index=0):
        self.release()
        self.cap = cv2.VideoCapture(int(index))
        self.timer.start()

    def open_video(self, path):
        self.release()
        self.cap = cv2.VideoCapture(path)
        self.timer.start()

    def open_image(self, path):
        self.release()
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is not None:
            self.frameCaptured.emit(img)

    def release(self):
        self.timer.stop()
        if self.cap is not None:
            try: self.cap.release()
            except: pass
        self.cap = None

    def _on_timeout(self):
        if self.cap is None:
            self.timer.stop(); return
        ok, frame = self.cap.read()
        if not ok or frame is None:
            self.release(); return
        self.frameCaptured.emit(frame)

# ----------------------------- Main Window ----------------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Haze-aware Face & Eye Detection")
        self.resize(1280, 760)

        self.params = Params()

        # central layout
        cw = QtWidgets.QWidget(); self.setCentralWidget(cw)
        hbox = QtWidgets.QHBoxLayout(cw)

        # views
        viewWrap = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(viewWrap)
        self.viewOrig = ImageView(640,360)
        self.viewProc = ImageView(640,360)
        grid.addWidget(QtWidgets.QLabel("Original"), 0,0)
        grid.addWidget(QtWidgets.QLabel("Processed"), 0,1)
        grid.addWidget(self.viewOrig, 1,0)
        grid.addWidget(self.viewProc, 1,1)

        # controls
        self.controls = ControlPanel(self.params)

        hbox.addWidget(viewWrap, 3)
        hbox.addWidget(self.controls, 2)

        # log / status
        self.logEdit = QtWidgets.QTextEdit(); self.logEdit.setReadOnly(True); self.logEdit.setFixedHeight(140)
        hbox2 = QtWidgets.QVBoxLayout(); hbox2.addStretch(1); hbox2.addWidget(self.logEdit)
        hbox.addLayout(hbox2, 0)

        self._statusLeft = QtWidgets.QLabel("Ready")
        self._statusRight = QtWidgets.QLabel("")
        self.statusBar().addWidget(self._statusLeft, 1)
        self.statusBar().addPermanentWidget(self._statusRight, 1)

        # menu
        self._make_menu()

        # source + worker
        self.source = VideoSource()
        self.source.frameCaptured.connect(self._on_frame_from_source)

        self.worker = Worker(self.params)
        self.workerThread = QtCore.QThread(self)
        self.worker.moveToThread(self.workerThread)
        self.workerThread.started.connect(self.worker.start)
        self.source.frameCaptured.connect(self.worker.enqueue)
        self.worker.frameReady.connect(self._on_processed)
        self.worker.logReady.connect(self._append_log)

        self.workerThread.start()

        # kernel dialog (lazy)
        self.kernelDlg = None

        # btns
        self.controls.btnWebcam.clicked.connect(self._on_webcam)
        self.controls.btnVideo.clicked.connect(self._on_video)
        self.controls.btnImage.clicked.connect(self._on_image)
        self.controls.btnStop.clicked.connect(self._on_stop)

        # Drag&Drop on original view
        self.viewOrig.setAcceptDrops(True)
        self.viewOrig.dragEnterEvent = self._drag_enter
        self.viewOrig.dropEvent = self._drop_event

        # graceful exit hooks
        QtWidgets.QApplication.instance().aboutToQuit.connect(self._shutdown_threads)
        import atexit; atexit.register(self._shutdown_threads)

        self._lastProc = None
        self._log_path = os.path.abspath("drowsy_log.txt")

    # ---------------- Menu ----------------
    def _make_menu(self):
        mb = self.menuBar()
        mFile = mb.addMenu("&File")
        actQuit = mFile.addAction("Quit")
        actQuit.triggered.connect(lambda: QtWidgets.QApplication.quit())

        mTools = mb.addMenu("&Tools")
        actKernel = mTools.addAction("Custom Kernel Editor…")
        actKernel.triggered.connect(self._open_kernel_dialog)

        mHelp = mb.addMenu("&Help")
        actAbout = mHelp.addAction("About")
        actAbout.triggered.connect(self._about)

    def _open_kernel_dialog(self):
        if self.kernelDlg is None:
            self.kernelDlg = KernelDialog(self._get_preview_frame)
            self.kernelDlg.kernelApplied.connect(self._apply_kernel)
        self.kernelDlg.show()
        self.kernelDlg.raise_()
        self.kernelDlg.activateWindow()

    def _get_preview_frame(self):
        return self._lastProc if self._lastProc is not None else None

    @QtCore.Slot(np.ndarray, bool)
    def _apply_kernel(self, k: np.ndarray, norm: bool):
        self.params.custom_kernel = k
        self.params.custom_norm = bool(norm)
        self.params.changed.emit()

    def _about(self):
        QtWidgets.QMessageBox.information(self, "About",
            "Haze-aware Face & Eye Detection\n"
            "PySide6 + OpenCV + MediaPipe FaceMesh\n"
            "EAR-based Drowsy Alarm\n"
            "© 2025 귀요미 후배가 오빠 사랑으로 제작 💙")

    # ------------- DnD -------------
    def _drag_enter(self, e: QtGui.QDragEnterEvent):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def _drop_event(self, e: QtGui.QDropEvent):
        for url in e.mimeData().urls():
            path = url.toLocalFile()
            if not path: continue
            ext = os.path.splitext(path)[1].lower()
            if ext in [".png",".jpg",".jpeg",".bmp",".tif",".tiff"]:
                self._open_image_path(path); break
            if ext in [".mp4",".avi",".mkv",".mov"]:
                self._open_video_path(path); break

    # ------------- Source handlers -------------
    def _on_webcam(self):
        idx, ok = QtWidgets.QInputDialog.getInt(self, "Webcam Index", "Camera index:", 0, 0, 16, 1)
        if ok:
            self._statusLeft.setText(f"Webcam: {idx}")
            self.source.open_webcam(idx)

    def _on_video(self):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open Video", "", "Video Files (*.mp4 *.avi *.mkv *.mov);;All Files (*)")
        if p: self._open_video_path(p)

    def _open_video_path(self, p):
        self._statusLeft.setText(f"Video: {os.path.basename(p)}")
        self.source.open_video(p)

    def _on_image(self):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp);;All Files (*)")
        if p: self._open_image_path(p)

    def _open_image_path(self, p):
        self._statusLeft.setText(f"Image: {os.path.basename(p)}")
        self.source.open_image(p)

    def _on_stop(self):
        self._statusLeft.setText("Stopped")
        self.source.release()

    # ------------- Frame slots -------------
    @QtCore.Slot(np.ndarray)
    def _on_frame_from_source(self, frame):
        self.viewOrig.setImage(frame)

    @QtCore.Slot(np.ndarray, np.ndarray, dict)
    def _on_processed(self, orig, proc, stats):
        self._lastProc = proc
        self.viewProc.setImage(proc)
        self._statusRight.setText(f"EAR: {stats.get('ear','-')}")

    @QtCore.Slot(str)
    def _append_log(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        self.logEdit.append(line)
        try:
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(line+"\n")
        except Exception:
            pass

    # ------------- Safe shutdown -------------
    def _shutdown_threads(self):
        if getattr(self, "_shutdown_done", False): return
        self._shutdown_done = True
        # disconnect & stop source
        try:
            self.source.frameCaptured.disconnect(self._on_frame_from_source)
        except Exception:
            pass
        try:
            self.source.frameCaptured.disconnect(self.worker.enqueue)
        except Exception:
            pass
        try:
            self.source.release()
        except Exception:
            pass
        # stop worker in its thread (blocking)
        try:
            QtCore.QMetaObject.invokeMethod(self.worker, "stop",
                QtCore.Qt.BlockingQueuedConnection)
        except Exception:
            pass
        # close thread
        try:
            self.workerThread.requestInterruption()
            self.workerThread.quit()
            if not self.workerThread.wait(3000):
                self.workerThread.terminate()
                self.workerThread.wait()
        except Exception:
            pass
        # delete later
        try: self.worker.deleteLater()
        except: pass
        try: self.workerThread.deleteLater()
        except: pass

    def closeEvent(self, e: QtGui.QCloseEvent) -> None:
        try:
            self._shutdown_threads()
        finally:
            super().closeEvent(e)

# ----------------------------- App entry ------------------------------
def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
