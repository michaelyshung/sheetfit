; Inno Setup script: wraps the PyInstaller folder (dist\sheetfit) into
; sheetfit-<version>-windows-setup.exe.
;   iscc /DAppVersion=0.6.0 packaging\sheetfit.iss
; Installs for the current user without admin rights by default (the user
; can choose all users). Adds a Start menu entry, an optional desktop icon,
; and optionally puts the `sheetfit` command-line tool on PATH.

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif

[Setup]
AppId={{6C1F0E8A-5B7D-4E7A-9F0B-3A2E5D8C9B41}
AppName=sheetfit
AppVersion={#AppVersion}
AppPublisher=michaelyshung
AppPublisherURL=https://github.com/michaelyshung/sheetfit
DefaultDirName={autopf}\sheetfit
DefaultGroupName=sheetfit
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=..\dist
OutputBaseFilename=sheetfit-{#AppVersion}-windows-setup
SetupIconFile=icon.ico
UninstallDisplayIcon={app}\sheetfit-app.exe
LicenseFile=..\LICENSE
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
ChangesEnvironment=yes

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop icon"; GroupDescription: "Additional icons:"; Flags: unchecked
Name: "addtopath"; Description: "Add the sheetfit &command-line tool to PATH"; GroupDescription: "Command line:"; Flags: unchecked

[Files]
Source: "..\dist\sheetfit\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\sheetfit"; Filename: "{app}\sheetfit-app.exe"
Name: "{autodesktop}\sheetfit"; Filename: "{app}\sheetfit-app.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\sheetfit-app.exe"; Description: "Open sheetfit"; Flags: nowait postinstall skipifsilent

[Code]
function EnvRoot: Integer;
begin
  if IsAdminInstallMode then Result := HKEY_LOCAL_MACHINE else Result := HKEY_CURRENT_USER;
end;

function EnvKey: String;
begin
  if IsAdminInstallMode then
    Result := 'SYSTEM\CurrentControlSet\Control\Session Manager\Environment'
  else
    Result := 'Environment';
end;

procedure AddToPath(Dir: String);
var Path: String;
begin
  if not RegQueryStringValue(EnvRoot, EnvKey, 'Path', Path) then Path := '';
  if Pos(';' + Uppercase(Dir) + ';', ';' + Uppercase(Path) + ';') > 0 then exit;
  if (Path <> '') and (Copy(Path, Length(Path), 1) <> ';') then Path := Path + ';';
  RegWriteExpandStringValue(EnvRoot, EnvKey, 'Path', Path + Dir);
end;

procedure RemoveFromPath(Dir: String);
var Path: String; P: Integer;
begin
  if not RegQueryStringValue(EnvRoot, EnvKey, 'Path', Path) then exit;
  Path := ';' + Path + ';';
  P := Pos(';' + Uppercase(Dir) + ';', Uppercase(Path));
  if P = 0 then exit;
  Delete(Path, P, Length(Dir) + 1);
  Path := Copy(Path, 2, Length(Path) - 2);
  RegWriteExpandStringValue(EnvRoot, EnvKey, 'Path', Path);
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if (CurStep = ssPostInstall) and WizardIsTaskSelected('addtopath') then
    AddToPath(ExpandConstant('{app}'));
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if CurUninstallStep = usPostUninstall then
    RemoveFromPath(ExpandConstant('{app}'));
end;
