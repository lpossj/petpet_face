# 上传到 GitHub

## 1. 上传源代码

先在 GitHub 网站创建空仓库 `petpet_face`。不要勾选自动创建 README、`.gitignore` 或 License。

在 PowerShell 执行：

```powershell
cd E:\Github\petpet_face\source
git init
git add .
git commit -m "Initial release"
git branch -M main
git remote add origin https://github.com/<你的用户名>/petpet_face.git
git push -u origin main
```

将 `<你的用户名>` 换成 GitHub 用户名。登录时按 GitHub 提示使用浏览器授权或 Personal Access Token，不能使用账户密码。

## 2. 上传 Release

1. 打开仓库主页。
2. 进入右侧 **Releases**，点击 **Draft a new release**。
3. 点击 **Choose a tag**，输入 `v0.0.1`，选择创建新标签。
4. 标题输入 `PetPet Face v0.0.1`。
5. 上传以下文件：
   - `E:\Github\petpet_face\release\PetPet-Face-v0.0.1-Windows-x64.zip`
   - `E:\Github\petpet_face\installer\output\PetPet-Face-Setup-v0.0.1-x64.exe`
   - `E:\Github\petpet_face\release\SHA256SUMS.txt`
6. 说明中写明默认使用 CPU；NVIDIA GPU 用户可额外下载现有可选组件：
   `https://github.com/lpossj/VideoPrivacy_optionalGPUcomponents/releases/tag/v0.0.1`
7. 点击 **Publish release**。

源码目录不要放 Release ZIP、DLL、FFmpeg 或 `cpp/build`。这些大文件只上传到 Releases。
