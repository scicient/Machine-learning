# -*- coding: utf-8 -*-
"""
ORL人脸识别交互GUI — ResNet18深度学习模型。
上传任意人脸图片（pgm/jpg/png/bmp），输出Top-3识别预测结果。
布局：左侧30%功能栏 + 右侧70%预览&结果（上下均分）。
"""

import os, sys, io, threading

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import numpy as np
import torch, torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageTk

# ==================== ResNet18 模型定义 ====================

IMG_SIZE = 112
NUM_CLASSES = 40
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class ResNet18(nn.Module):
    def __init__(self, num_classes=40, in_channels=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = self._make(64, 64, 2, 1)
        self.layer2 = self._make(64, 128, 2, 2)
        self.layer3 = self._make(128, 256, 2, 2)
        self.layer4 = self._make(256, 512, 2, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make(self, in_c, out_c, blocks, stride):
        ds = None
        if stride != 1 or in_c != out_c:
            ds = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride, bias=False),
                nn.BatchNorm2d(out_c))
        ly = [ResidualBlock(in_c, out_c, stride, ds)]
        for _ in range(1, blocks):
            ly.append(ResidualBlock(out_c, out_c))
        return nn.Sequential(*ly)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        x = self.avgpool(x)
        return self.fc(torch.flatten(x, 1))


TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

# ==================== 配色方案 ====================

C_PRIMARY   = "#4f46e5"
C_PRIMARY_D = "#4338ca"
C_ACCENT    = "#f59e0b"
C_SUCCESS   = "#10b981"
C_DANGER    = "#ef4444"
C_SIDEBAR   = "#1e1b4b"
C_BG        = "#f1f5f9"
C_CARD      = "#ffffff"
C_TEXT      = "#1e293b"
C_TEXT_L    = "#64748b"
C_BORDER    = "#e2e8f0"
C_INNER_BG  = "#f8fafc"


class RoundedButton(tk.Canvas):
    """圆角按钮（Canvas实现，鼠标悬浮变色）"""
    def __init__(self, parent, text, command, bg=C_PRIMARY, hover=C_PRIMARY_D,
                 width=120, height=34, font_size=10, bold=False):
        super().__init__(parent, width=width, height=height,
                         bg=C_CARD, highlightthickness=0, cursor="hand2")
        self.command = command
        self.btn_bg = bg
        self.btn_hover = hover
        self.btn_text = text
        self.font = ("微软雅黑", font_size, "bold" if bold else "normal")
        self._draw(bg)
        self.bind("<Enter>", lambda e: self._draw(hover))
        self.bind("<Leave>", lambda e: self._draw(bg))
        self.bind("<Button-1>", lambda e: command())

    def _draw(self, color):
        self.delete("all")
        w, h = int(self["width"]), int(self["height"])
        r = 8
        self.create_polygon(
            r, 0, w - r, 0, w - r, 0, w, 0, w, r,
            w, h - r, w, h, w - r, h, r, h, 0, h, 0, h - r,
            0, r, 0, 0, r, 0,
            smooth=True, fill=color, outline="")
        self.create_text(w // 2, h // 2, text=self.btn_text,
                          font=self.font, fill="white")


# ==================== 主GUI类 ====================

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ORL人脸识别系统 — 残差网络18层(ResNet18)深度学习")
        w_s, h_s = root.winfo_screenwidth(), root.winfo_screenheight()
        self._ww, self._wh = 1000, 750
        x = (w_s - self._ww) // 2
        y = (h_s - self._wh) // 2
        self.root.geometry(f"{self._ww}x{self._wh}+{x}+{y}")
        self.root.minsize(900, 650)
        self.root.configure(bg=C_BG)

        self.model = None
        self.current_image = None
        self._result_shown = False      # 标记是否已显示识别结果
        self._project_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path_var = tk.StringVar(value=os.path.join(
            self._project_dir, "实验结果", "最优模型_7比3划分.pth"))

        self._build_ui()
        self._load_model()

    # ==================== 全局布局 ====================

    def _build_ui(self):
        # ——— 顶部导航栏 ———
        topbar = tk.Frame(self.root, bg=C_SIDEBAR, height=48)
        topbar.pack(fill=tk.X)
        topbar.pack_propagate(False)

        tk.Label(topbar, text="●", font=("微软雅黑", 16),
                 fg=C_ACCENT, bg=C_SIDEBAR).pack(side=tk.LEFT, padx=(16, 6))
        tk.Label(topbar, text="ORL 人脸识别系统",
                 font=("微软雅黑", 13, "bold"), fg="white", bg=C_SIDEBAR).pack(
            side=tk.LEFT, pady=12)
        tk.Label(topbar, text="ResNet18  ·  PyTorch  ·  40类分类",
                 font=("微软雅黑", 8), fg="#94a3b8", bg=C_SIDEBAR).pack(
            side=tk.RIGHT, padx=16, pady=15)

        # ——— 左右分栏 PanedWindow ———
        body_pane = tk.PanedWindow(self.root, orient=tk.HORIZONTAL,
                                    bg=C_BG, sashwidth=4, sashrelief=tk.FLAT)
        body_pane.pack(fill=tk.BOTH, expand=True, padx=14, pady=(12, 0))

        # ---- 左侧面板 ----
        left_frame = tk.Frame(body_pane, bg=C_CARD, bd=0,
                               highlightthickness=1, highlightbackground=C_BORDER)
        body_pane.add(left_frame, minsize=260, width=300)

        self._build_left_panel(left_frame)

        # ---- 右侧面板 ----
        right_frame = tk.Frame(body_pane, bg=C_BG, bd=0)
        body_pane.add(right_frame, minsize=400)

        self._build_right_panel(right_frame)

        # ——— 底部状态栏 ———
        bottombar = tk.Frame(self.root, bg=C_SIDEBAR, height=28)
        bottombar.pack(fill=tk.X, side=tk.BOTTOM)
        bottombar.pack_propagate(False)

        self.status_var = tk.StringVar(value="● 就绪 — 模型加载中...")
        tk.Label(bottombar, textvariable=self.status_var,
                 font=("微软雅黑", 8), fg="#94a3b8", bg=C_SIDEBAR,
                 anchor=tk.W, padx=12).pack(fill=tk.BOTH, pady=4)

        # 初始时把PanedWindow的sash调到30%位置
        self.root.update_idletasks()
        body_pane.sash_place(0, int(self._ww * 0.30), 0)

    # ==================== 左侧面板 ====================

    def _build_left_panel(self, parent):
        # 用Scrollable方式防内容溢出
        canvas = tk.Canvas(parent, bg=C_CARD, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=canvas.yview)
        scroll_frame = tk.Frame(canvas, bg=C_CARD)

        scroll_frame.bind("<Configure>",
                          lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scroll_frame, anchor=tk.NW,
                              tags="scroll_inner")
        canvas.configure(yscrollcommand=scrollbar.set)

        def _on_canvas_resize(event):
            canvas.itemconfig("scroll_inner", width=event.width)
        canvas.bind("<Configure>", _on_canvas_resize)

        # ---- 模型加载 ----
        self._section(scroll_frame, "模型加载")

        card = tk.Frame(scroll_frame, bg=C_INNER_BG, bd=0)
        card.pack(fill=tk.X, padx=14, pady=(2, 6))

        tk.Label(card, text="权重文件", font=("微软雅黑", 9),
                 fg=C_TEXT_L, bg=C_INNER_BG).pack(anchor=tk.W, padx=10, pady=(8, 0))

        eb = tk.Frame(card, bg="white", bd=1, relief=tk.SOLID,
                      highlightbackground=C_BORDER)
        eb.pack(fill=tk.X, padx=10, pady=(2, 8))
        tk.Entry(eb, textvariable=self.model_path_var,
                 font=("Consolas", 7), fg=C_TEXT_L, bg="white",
                 relief=tk.FLAT, state="readonly", bd=0).pack(fill=tk.X, padx=5, pady=3)

        btn_row = tk.Frame(card, bg=C_INNER_BG)
        btn_row.pack(fill=tk.X, padx=10)
        RoundedButton(btn_row, "重新加载", self._load_model,
                       bg=C_PRIMARY, hover=C_PRIMARY_D,
                       width=90, height=28, font_size=9).pack(side=tk.LEFT, padx=(0, 6))
        RoundedButton(btn_row, "浏览选择", self._browse_model,
                       bg="#64748b", hover="#475569",
                       width=90, height=28, font_size=9).pack(side=tk.LEFT)

        self.model_info_text = tk.StringVar(value="等待加载模型...")
        tk.Label(card, textvariable=self.model_info_text,
                 font=("微软雅黑", 8), fg=C_TEXT_L, bg=C_INNER_BG,
                 justify=tk.LEFT, wraplength=240).pack(
            anchor=tk.W, padx=10, pady=(6, 10))

        # ---- 分隔 ----
        ttk.Separator(scroll_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=14, pady=2)

        # ---- 图片操作 ----
        self._section(scroll_frame, "图片操作")

        card2 = tk.Frame(scroll_frame, bg=C_INNER_BG, bd=0)
        card2.pack(fill=tk.X, padx=14, pady=(2, 6))

        btn_row2 = tk.Frame(card2, bg=C_INNER_BG)
        btn_row2.pack(fill=tk.X, padx=10, pady=10)
        RoundedButton(btn_row2, "选择图片", self._select_image,
                       bg=C_SUCCESS, hover="#059669",
                       width=100, height=32, font_size=10).pack(side=tk.LEFT, padx=(0, 6))
        RoundedButton(btn_row2, "开始识别", self._predict,
                       bg=C_DANGER, hover="#dc2626",
                       width=100, height=32, font_size=10, bold=True).pack(side=tk.LEFT)

        self.img_path_var = tk.StringVar(value="尚未选择图片")
        tk.Label(card2, textvariable=self.img_path_var,
                 font=("微软雅黑", 8), fg=C_TEXT_L, bg=C_INNER_BG,
                 wraplength=240).pack(anchor=tk.W, padx=10, pady=(0, 10))

        # ---- 分隔 ----
        ttk.Separator(scroll_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=14, pady=2)

        # ---- 使用说明 ----
        self._section(scroll_frame, "使用说明")
        tips = ("① 请先运行训练脚本生成完整版模型\n"
                "② 点击「选择图片」上传人脸图片\n"
                "③ 点击「开始识别」获取预测结果\n"
                "④ 右侧展示图片预览与概率柱状图\n"
                "⑤ 支持 pgm / jpg / png / bmp 格式\n"
                "⑥ 分类范围 s1 ~ s40，共 40 类")
        tk.Label(scroll_frame, text=tips, font=("微软雅黑", 8),
                 fg=C_TEXT_L, bg=C_CARD, justify=tk.LEFT,
                 wraplength=260).pack(anchor=tk.W, padx=14, pady=(4, 18))

        # 左侧面板滚动
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def _section(self, parent, text):
        """统一小节标题"""
        tk.Label(parent, text="▎" + text, font=("微软雅黑", 11, "bold"),
                 fg=C_TEXT, bg=C_CARD).pack(anchor=tk.W, padx=14, pady=(10, 4))

    # ==================== 右侧面板 ====================

    def _build_right_panel(self, parent):
        """右侧：grid划分，预览65%、结果35%"""
        parent.grid_rowconfigure(0, weight=65)   # 预览占大份
        parent.grid_rowconfigure(1, weight=35)   # 结果占小份
        parent.grid_columnconfigure(0, weight=1)

        # ---- 上半：图片预览 ----
        top_frame = tk.Frame(parent, bg=C_CARD, bd=0,
                              highlightthickness=1, highlightbackground=C_BORDER)
        top_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 3))
        top_frame.grid_rowconfigure(0, weight=0)   # 标题
        top_frame.grid_rowconfigure(1, weight=1)   # 画布填满
        top_frame.grid_columnconfigure(0, weight=1)

        self._section_grid(top_frame, "图片预览", row=0)
        self._build_preview(top_frame, row=1)

        # ---- 下半：识别结果 ----
        bot_frame = tk.Frame(parent, bg=C_CARD, bd=0,
                              highlightthickness=1, highlightbackground=C_BORDER)
        bot_frame.grid(row=1, column=0, sticky="nsew", pady=(3, 0))
        bot_frame.grid_rowconfigure(0, weight=0)   # 标题
        bot_frame.grid_rowconfigure(1, weight=0)   # 文字（固定高）
        bot_frame.grid_rowconfigure(2, weight=1)   # 柱状图填满剩余
        bot_frame.grid_columnconfigure(0, weight=1)

        self._section_grid(bot_frame, "识别结果", row=0)
        self._build_result_text(bot_frame, row=1)
        self._build_result_chart(bot_frame, row=2)

    def _section_grid(self, parent, text, row):
        """grid版小节标题"""
        f = tk.Frame(parent, bg=C_CARD)
        f.grid(row=row, column=0, sticky="ew", padx=14, pady=(10, 2))
        tk.Label(f, text="▎" + text, font=("微软雅黑", 11, "bold"),
                 fg=C_TEXT, bg=C_CARD).pack(anchor=tk.W)

    # ---- 图片预览模块 ----

    def _build_preview(self, parent, row):
        """预览区：Canvas铺满，给定基础尺寸"""
        self.preview_canvas = tk.Canvas(parent, bg="#f1f5f9",
                                         height=300, width=400,
                                         highlightthickness=1,
                                         highlightbackground=C_BORDER)
        self.preview_canvas.grid(row=row, column=0, sticky="nsew",
                                  padx=14, pady=(2, 10))
        self.canvas_image_id = None

    def _draw_preview_placeholder(self, _event=None):
        """在预览Canvas上绘制空状态占位文字"""
        self.preview_canvas.delete("placeholder")
        w = self.preview_canvas.winfo_width()
        h = self.preview_canvas.winfo_height()
        if w > 20 and h > 20:
            self.preview_canvas.create_text(
                w // 2, h // 2, tags="placeholder",
                text="📷\n\n暂无图片\n请点击「选择图片」上传",
                font=("微软雅黑", 12), fill="#94a3b8", justify=tk.CENTER)

    # ---- 识别结果：文字 ----

    def _build_result_text(self, parent, row):
        """预测文字区"""
        txt = tk.Frame(parent, bg=C_INNER_BG, bd=0)
        txt.grid(row=row, column=0, sticky="ew", padx=14, pady=(2, 2))
        txt.grid_columnconfigure(0, weight=1)

        self.pred_label_var = tk.StringVar(value="等待图片识别...")
        tk.Label(txt, textvariable=self.pred_label_var,
                 font=("微软雅黑", 12, "bold"), fg=C_TEXT, bg=C_INNER_BG
                 ).pack(anchor=tk.W, padx=12, pady=(6, 1))

        self.conf_var = tk.StringVar(value="")
        self.conf_label = tk.Label(txt, textvariable=self.conf_var,
                                    font=("微软雅黑", 13, "bold"),
                                    fg=C_PRIMARY, bg=C_INNER_BG)
        self.conf_label.pack(anchor=tk.W, padx=12, pady=(0, 4))

    # ---- 识别结果：柱状图 ----

    def _build_result_chart(self, parent, row):
        """柱状图区：Canvas铺满，给定基础尺寸"""
        self.bar_canvas = tk.Canvas(parent, bg=C_INNER_BG,
                                     height=120, width=400,
                                     highlightthickness=0)
        self.bar_canvas.grid(row=row, column=0, sticky="nsew",
                              padx=14, pady=(0, 10))

    def _draw_empty_chart(self):
        """绘制柱状图空状态"""
        self.bar_canvas.delete("all")
        w = self.bar_canvas.winfo_width()
        h = self.bar_canvas.winfo_height()
        if w > 30 and h > 30:
            self.bar_canvas.create_text(
                w // 2, h // 2,
                text="Top‑3 概率柱状图将在识别后显示",
                font=("微软雅黑", 10), fill="#94a3b8")

    # ==================== 业务逻辑 ====================

    def _load_model(self):
        candidates = [
            self.model_path_var.get(),
            os.path.join(self._project_dir, "实验结果", "最优模型_7比3划分.pth"),
            os.path.join(self._project_dir, "实验结果", "最优模型_5比5划分.pth"),
        ]
        model_path = next((p for p in candidates if os.path.exists(p)), None)

        if model_path is None:
            self.model_info_text.set(
                "⚠ 未找到完整版模型文件\n"
                "请先运行训练脚本生成 *_完整版.pth")
            self.status_var.set("● 模型加载失败 — 文件不存在")
            return

        def _load():
            try:
                self.status_var.set("● 正在加载模型...")
                m = torch.load(model_path, map_location=DEVICE, weights_only=False)
                m.eval()
                self.root.after(0, lambda: self._on_model_ok(model_path, m))
            except Exception as e:
                self.root.after(0, lambda: self._on_model_err(str(e)))

        threading.Thread(target=_load, daemon=True).start()

    def _on_model_ok(self, path, model):
        self.model = model
        mb = os.path.getsize(path) / 1024 / 1024
        self.model_info_text.set(
            f"✔ 模型已就绪\n"
            f"  ResNet18 · 约1119万参数\n"
            f"  40类（s1~s40）\n"
            f"  {os.path.basename(path)}（{mb:.1f}MB）")
        self.status_var.set("● 就绪 — 请选择图片")

    def _on_model_err(self, err):
        self.model_info_text.set(f"✘ 加载失败：{err}")
        self.status_var.set("● 模型加载失败")

    def _browse_model(self):
        path = filedialog.askopenfilename(
            title="选择完整版模型文件",
            filetypes=[("PyTorch完整模型", "*.pth"), ("所有文件", "*.*")])
        if path:
            self.model_path_var.set(path)
            self._load_model()

    def _select_image(self):
        path = filedialog.askopenfilename(
            title="选择人脸图片",
            filetypes=[("图片文件", "*.pgm *.jpg *.jpeg *.png *.bmp"),
                       ("所有文件", "*.*")])
        if not path:
            return
        try:
            img = Image.open(path)
            self.img_path_var.set(f"已选择：{os.path.basename(path)}")
            self.current_image = np.array(img.convert("L"), dtype=np.uint8)

            # 预览
            self.preview_canvas.delete("placeholder")
            if self.canvas_image_id:
                self.preview_canvas.delete(self.canvas_image_id)

            pw = max(self.preview_canvas.winfo_width() - 8, 60)
            ph = max(self.preview_canvas.winfo_height() - 8, 60)
            disp = img.copy()
            disp.thumbnail((pw, ph), Image.LANCZOS)
            tk_img = ImageTk.PhotoImage(disp)

            cx = (pw - disp.width) // 2 + 4
            cy = (ph - disp.height) // 2 + 4
            self.canvas_image_id = self.preview_canvas.create_image(
                cx, cy, anchor=tk.NW, image=tk_img)
            self.preview_canvas.image = tk_img

            self.pred_label_var.set("等待识别...")
            self.conf_var.set("")
            self._result_shown = False
            self.conf_label.configure(fg=C_PRIMARY)
            self._draw_empty_chart()
            self.status_var.set(f"● 图片已加载 — {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("图片加载失败", f"{e}")

    def _predict(self):
        if self.model is None:
            messagebox.showwarning("未就绪", "请先等待模型加载完成！")
            return
        if self.current_image is None:
            messagebox.showwarning("未选择图片", "请先选择一张人脸图片！")
            return

        self.status_var.set("● 正在识别...")
        self.pred_label_var.set("识别中，请稍候...")

        def _run():
            try:
                t = TRANSFORM(self.current_image).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    out = torch.softmax(self.model(t), dim=1).cpu().numpy()[0]
                idx3 = np.argsort(out)[::-1][:3]
                prob3 = out[idx3]
                self.root.after(0, lambda: self._show_result(idx3, prob3))
            except Exception as e:
                self.root.after(0, lambda: self._on_err(str(e)))

        threading.Thread(target=_run, daemon=True).start()

    def _show_result(self, idx3, prob3):
        self._result_shown = True
        best = idx3[0]
        conf = prob3[0]

        self.pred_label_var.set(f"预测结果：s{best + 1}（类别 {best + 1} / 40）")

        if conf > 0.85:
            badge, badge_c = "● 高可信度", C_SUCCESS
        elif conf > 0.50:
            badge, badge_c = "● 一般可信度", C_ACCENT
        else:
            badge, badge_c = "● 低可信度", C_DANGER

        self.conf_var.set(f"置信度  {conf * 100:.1f}%    {badge}")
        self.conf_label.configure(fg=badge_c)

        self._draw_bars(idx3, prob3)
        self.status_var.set(f"● 识别完成 — s{best + 1}（{conf * 100:.1f}%）")

    def _draw_bars(self, idx3, prob3):
        self.bar_canvas.delete("all")
        w = self.bar_canvas.winfo_width()
        h = self.bar_canvas.winfo_height()
        if w < 40 or h < 40:
            return

        labels = [f"s{i + 1}" for i in idx3]
        colors = [C_PRIMARY, "#818cf8", "#c7d2fe"]
        bar_w = max(35, min(75, (w - 80) // 3))
        gap = (w - 3 * bar_w) // 4

        # 紧凑边距，适配较小画布
        top_m = 30
        bot_m = 22
        bar_area = max(20, h - top_m - bot_m)
        max_p = max(prob3)
        baseline = top_m + bar_area

        for i, (lb, p, c) in enumerate(zip(labels, prob3, colors)):
            bh = max(4, int((p / max_p) * bar_area))
            x = gap + i * (bar_w + gap)
            y0 = baseline - bh
            y1 = baseline
            mx = x + bar_w // 2

            self.bar_canvas.create_rectangle(x, y0, x + bar_w, y1,
                                              fill=c, outline="", width=0)
            # 百分数贴柱子顶部
            self.bar_canvas.create_text(mx, y0 - 8,
                                         text=f"{p * 100:.1f}%",
                                         font=("微软雅黑", 8, "bold"),
                                         fill=C_TEXT)
            # 标签贴柱子底部
            self.bar_canvas.create_text(mx, y1 + 10,
                                         text=lb,
                                         font=("微软雅黑", 9, "bold"),
                                         fill=C_TEXT)
            # 🏆 最佳标记
            if i == 0 and p > 0.5:
                self.bar_canvas.create_text(mx, y0 - 20,
                                             text="🏆最佳",
                                             font=("微软雅黑", 7, "bold"),
                                             fill=C_ACCENT)

    def _on_err(self, msg):
        self.pred_label_var.set("✘ 识别失败")
        self.conf_var.set("")
        self._draw_empty_chart()
        self.status_var.set("● 识别出错")
        messagebox.showerror("识别失败", f"{msg}")


# ==================== 启动 ====================

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)

    # 窗口首次显示后绘制占位文字
    def _init_placeholders():
        app._draw_preview_placeholder()
        app._draw_empty_chart()
    root.after(200, _init_placeholders)

    # 窗口缩放时刷新占位文字和预览图（不覆盖已显示的识别结果）
    def _on_resize(_event):
        if app.current_image is None:
            app._draw_preview_placeholder()
        if not app._result_shown:
            app._draw_empty_chart()
    root.bind("<Configure>", _on_resize)

    root.mainloop()
