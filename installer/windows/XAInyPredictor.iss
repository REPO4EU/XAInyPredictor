#define MyAppName "XAInyPredictor"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "REPO4EU Consortium"
#define MyAppURL "https://repo4.eu/"
#define MyAppExeName "XAInyPredictor.exe"

[Setup]
AppId={{6F8D6C10-6D5C-4D7E-8B92-5E2561F35E21}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName={localappdata}\Programs\{#MyAppName}
DefaultGroupName={#MyAppName}
OutputDir=..\..\dist-installer
OutputBaseFilename=XAInyPredictor-Setup-{#MyAppVersion}-Windows
Compression=lzma
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
SetupIconFile=assets\XAInyPredictor.ico
UninstallDisplayIcon={app}\XAInyPredictor.ico
WizardSmallImageFile=assets\wizard-small.png
WizardImageFile=assets\wizard-large.png
CloseApplications=yes
RestartApplications=no

[Files]
Source: "..\..\dist\XAInyPredictor\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "assets\XAInyPredictor.ico"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\XAInyPredictor.ico"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\XAInyPredictor.ico"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional icons:"; Flags: unchecked

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; Flags: nowait postinstall skipifsilent
