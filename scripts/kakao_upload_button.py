import threading
import tkinter as tk
from tkinter import filedialog, messagebox
from datetime import datetime
import os

from scripts.kakao_admin_uploader import upload_to_kakao


def run_upload(url, image_dir, title, headless, user_data_dir, profile_directory, driver_path, manual_login):
    try:
        # For "manual login" flow we can pass empty credentials.
        upload_to_kakao(
            url=url,
            email="",
            password="",
            image_dir=image_dir,
            headless=headless,
            user_data_dir=user_data_dir or None,
            profile_directory=profile_directory or None,
            driver_path=driver_path or None,
            title=title or None,
            manual_login=manual_login,
        )
        messagebox.showinfo("완료", "업로드가 완료되었습니다.")
    except Exception as e:
        messagebox.showerror("오류", f"업로드 중 오류가 발생했습니다:\n{e}")


def main():
    root = tk.Tk()
    root.title("Kakao 업로드 버튼")

    # URL
    tk.Label(root, text="글쓰기 URL").grid(row=0, column=0, sticky="e", padx=6, pady=6)
    url_var = tk.StringVar()
    tk.Entry(root, textvariable=url_var, width=60).grid(row=0, column=1, columnspan=2, sticky="we", padx=6)

    # Image Dir
    tk.Label(root, text="이미지 폴더").grid(row=1, column=0, sticky="e", padx=6, pady=6)
    img_var = tk.StringVar()
    tk.Entry(root, textvariable=img_var, width=50).grid(row=1, column=1, sticky="we", padx=6)
    def pick_dir():
        d = filedialog.askdirectory()
        if d:
            img_var.set(d)
    tk.Button(root, text="찾기", command=pick_dir).grid(row=1, column=2, padx=4)

    # Title
    tk.Label(root, text="제목").grid(row=2, column=0, sticky="e", padx=6, pady=6)
    title_var = tk.StringVar(value=datetime.now().strftime("%Y-%m-%d"))
    tk.Entry(root, textvariable=title_var, width=60).grid(row=2, column=1, columnspan=2, sticky="we", padx=6)

    # Options
    headless_var = tk.BooleanVar(value=False)
    manual_login_var = tk.BooleanVar(value=True)
    tk.Checkbutton(root, text="헤드리스(창 숨김)", variable=headless_var).grid(row=3, column=1, sticky="w", padx=6)
    tk.Checkbutton(root, text="수동 로그인(카카오 인증 직접 진행)", variable=manual_login_var).grid(row=3, column=2, sticky="w", padx=6)

    # User data dir
    tk.Label(root, text="Chrome User Data Dir").grid(row=4, column=0, sticky="e", padx=6, pady=6)
    udir_var = tk.StringVar(value=os.path.expanduser("~/.config/google-chrome"))
    tk.Entry(root, textvariable=udir_var, width=50).grid(row=4, column=1, sticky="we", padx=6)
    def pick_udir():
        d = filedialog.askdirectory()
        if d:
            udir_var.set(d)
    tk.Button(root, text="찾기", command=pick_udir).grid(row=4, column=2, padx=4)

    # Profile directory
    tk.Label(root, text="Profile 디렉터리명").grid(row=5, column=0, sticky="e", padx=6, pady=6)
    prof_var = tk.StringVar(value="Default")
    tk.Entry(root, textvariable=prof_var, width=60).grid(row=5, column=1, columnspan=2, sticky="we", padx=6)

    # Driver path
    tk.Label(root, text="ChromeDriver 경로").grid(row=6, column=0, sticky="e", padx=6, pady=6)
    dpath_var = tk.StringVar()
    tk.Entry(root, textvariable=dpath_var, width=50).grid(row=6, column=1, sticky="we", padx=6)
    def pick_driver():
        f = filedialog.askopenfilename()
        if f:
            dpath_var.set(f)
    tk.Button(root, text="찾기", command=pick_driver).grid(row=6, column=2, padx=4)

    # Upload button
    def on_upload():
        url = url_var.get().strip()
        image_dir = img_var.get().strip()
        title = title_var.get().strip()
        user_data_dir = udir_var.get().strip()
        profile_dir = prof_var.get().strip()
        driver_path = dpath_var.get().strip()

        if not url:
            messagebox.showwarning("확인", "글쓰기 URL을 입력하세요.")
            return
        if not image_dir:
            messagebox.showwarning("확인", "이미지 폴더를 선택하세요.")
            return

        threading.Thread(
            target=run_upload,
            args=(
                url,
                image_dir,
                title,
                headless_var.get(),
                user_data_dir,
                profile_dir,
                driver_path,
                manual_login_var.get(),
            ),
            daemon=True,
        ).start()

    tk.Button(root, text="업로드", command=on_upload, height=2).grid(row=7, column=0, columnspan=3, sticky="we", padx=6, pady=12)

    for i in range(3):
        root.grid_columnconfigure(i, weight=1)

    root.mainloop()


if __name__ == "__main__":
    main()

