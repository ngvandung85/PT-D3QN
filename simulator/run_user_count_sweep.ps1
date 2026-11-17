param(
    [string]$Output = "results_user_count_sweep",
    [string]$Seeds = "7,17,27,37,47,57,67,77,87,97,107,117,127,137,147,157,167,177,187,197",
    [int]$TrainEpisodes = 3000,
    [int]$EvalEpisodes = 100,
    [int]$ReferenceCalibrationEpisodes = 50,
    [int]$TrainingProfileCount = 20,
    [double]$TrainingReferenceJitter = 0.20,
    [double]$InitialRadioEnergyJ = 2.0,
    [int]$ReplaySize = 50000,
    [int]$UpdateFrequency = 4,
    [switch]$Quick,
    [switch]$AllowExploratorySeedCount,
    [switch]$NoStepDetails,
    [switch]$SkipVerification
)

$ErrorActionPreference = "Stop"

if (Get-Command py -ErrorAction SilentlyContinue) {
    $PythonExe = "py"
}
elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $PythonExe = "python"
}
else {
    throw "Python was not found. Install Python and disable the Microsoft Store alias if necessary."
}

$ProgramPath = Join-Path $PSScriptRoot "pt_d3qn_uav.py"
$TestPath = Join-Path $PSScriptRoot "test_pt_d3qn_uav.py"
$AuditPath = Join-Path $PSScriptRoot "verify_algorithms_independently.py"
$ValidatorPath = Join-Path $PSScriptRoot "validate_results.py"
$OutputPath = [System.IO.Path]::GetFullPath((Join-Path (Get-Location) $Output))
$SeedValues = @(
    $Seeds.Split(',') |
        ForEach-Object { $_.Trim() } |
        Where-Object { $_ -ne "" }
)
if (($SeedValues | Select-Object -Unique).Count -ne $SeedValues.Count) {
    throw "Seeds must be unique: $Seeds"
}
if (-not $Quick -and -not $AllowExploratorySeedCount -and $SeedValues.Count -lt 10) {
    throw "Final runs require at least 10 independent seeds. Use -Quick or -AllowExploratorySeedCount only for pilot work."
}

if (Test-Path $OutputPath) {
    $ExistingItems = @(Get-ChildItem -LiteralPath $OutputPath -Force)
    if ($ExistingItems.Count -gt 0) {
        throw "Output folder is not empty: $OutputPath. Choose a new -Output value."
    }
}

if (-not $SkipVerification) {
    Push-Location $PSScriptRoot
    try {
        & $PythonExe -m unittest -v (Split-Path $TestPath -Leaf)
        if ($LASTEXITCODE -ne 0) {
            throw "Unit tests failed with exit code $LASTEXITCODE"
        }
        & $PythonExe $AuditPath
        if ($LASTEXITCODE -ne 0) {
            throw "Independent numerical audit failed with exit code $LASTEXITCODE"
        }
    }
    finally {
        Pop-Location
    }
}

$RunArguments = @(
    $ProgramPath,
    "--user-counts", "5,15,30,45,60",
    "--seeds", $Seeds,
    "--random-pt-profiles",
    "--reference-mode", "random_policy_calibration",
    "--training-profile-count", $TrainingProfileCount,
    "--training-reference-jitter", $TrainingReferenceJitter,
    "--initial-radio-energy-j", $InitialRadioEnergyJ,
    "--replay-size", $ReplaySize,
    "--update-frequency", $UpdateFrequency,
    "--output", $OutputPath
)

if ($Quick) {
    $RunArguments += "--quick"
}
# Do not pass the full-run defaults in Quick mode: Python's --quick values
# (8 train episodes, 3 evaluation episodes, 5 calibration episodes) must remain
# effective unless the caller explicitly overrides an individual parameter.
if (-not $Quick -or $PSBoundParameters.ContainsKey("TrainEpisodes")) {
    $RunArguments += @("--train-episodes", $TrainEpisodes)
}
if (-not $Quick -or $PSBoundParameters.ContainsKey("EvalEpisodes")) {
    $RunArguments += @("--eval-episodes", $EvalEpisodes)
}
if (-not $Quick -or $PSBoundParameters.ContainsKey("ReferenceCalibrationEpisodes")) {
    $RunArguments += @(
        "--reference-calibration-episodes", $ReferenceCalibrationEpisodes
    )
}
if ($NoStepDetails) {
    $RunArguments += "--no-step-details"
}

Write-Host "Running PT-D3QN user-count sweep with $PythonExe"
Write-Host "Output: $OutputPath"
Write-Host "Seeds: $Seeds"
Write-Host "Training PT profiles per replicate: $TrainingProfileCount"

& $PythonExe @RunArguments
if ($LASTEXITCODE -ne 0) {
    throw "PT-D3QN sweep failed with exit code $LASTEXITCODE"
}

& $PythonExe $ValidatorPath $OutputPath
if ($LASTEXITCODE -ne 0) {
    throw "Result validation failed with exit code $LASTEXITCODE"
}
