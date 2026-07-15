#define MyAppName "PetPet Face"
#define MyAppVersion "0.0.1"
#define MyAppPublisher "lpossj"
#define MyAppExeName "PetPetFace.exe"

#define GpuPackageUrl "https://github.com/lpossj/VideoPrivacy_optionalGPUcomponents/releases/download/v0.0.1/VideoPrivacy-GPU-Runtime-v0.0.1-CUDA12-cuDNN9-x64.zip"
#define GpuPackageHash "7EFAE8571F4E216E2C1153617247A09CF8F2437D73141DE55354D3E92AFF2083"

[Setup]
AppId={{8F34F0A3-79E7-4F1B-9F9E-1BF15E6E77C2}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={localappdata}\Programs\PetPet Face
DefaultGroupName={#MyAppName}
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=..\..\installer\output
OutputBaseFilename=PetPet-Face-Setup-v{#MyAppVersion}-x64
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ArchiveExtraction=full
AlwaysShowComponentsList=yes
ShowLanguageDialog=no
DisableDirPage=no
AppendDefaultDirName=yes
UninstallDisplayIcon={app}\{#MyAppExeName}
VersionInfoVersion={#MyAppVersion}.0
VersionInfoCompany={#MyAppPublisher}
VersionInfoDescription=PetPet Face Installer
VersionInfoProductName={#MyAppName}
VersionInfoProductVersion={#MyAppVersion}

[Languages]
Name: "chinesesimp"; MessagesFile: "compiler:Languages\ChineseSimplified.isl"

[Types]
Name: "cpu"; Description: "标准安装（CPU，推荐）"
Name: "full"; Description: "完整安装（CPU + NVIDIA GPU）"
Name: "custom"; Description: "自定义安装"; Flags: iscustom

[Components]
Name: "core"; Description: "PetPet Face 核心程序、OpenCV、模型和 FFmpeg"; Types: cpu full custom; Flags: fixed
Name: "gpu"; Description: "NVIDIA GPU 加速组件（联网下载约 1.54 GB）"; Types: full

[Files]
Source: "..\..\release\PetPet-Face-v0.0.1-Windows-x64\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs; Components: core
Source: "{#GpuPackageUrl}"; DestName: "PetPet-Face-GPU-Runtime-CUDA12-cuDNN9-x64.zip"; DestDir: "{app}"; Hash: "{#GpuPackageHash}"; ExternalSize: 2_479_111_734; Flags: external download extractarchive recursesubdirs createallsubdirs ignoreversion; Components: gpu

[Dirs]
Name: "{app}\input"
Name: "{app}\output"

[Tasks]
Name: "desktopicon"; Description: "创建桌面快捷方式"; Flags: unchecked

[Icons]
Name: "{group}\PetPet Face"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"
Name: "{userdesktop}\PetPet Face"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "启动 PetPet Face"; WorkingDir: "{app}"; Flags: nowait postinstall skipifsilent
