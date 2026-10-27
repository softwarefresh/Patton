param([switch]$Stage)
$ErrorActionPreference = "Stop"
$projectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$sourceFiles = @(
    '.gitignore', 'PROGRESS.md', 'requirements.partner4090.txt',
    'src/setup_partner_4090.sh',
    'src/run_partner_teacher.sh',
    'src/run_partner_teacher_calibration.sh',
    'data_pipeline/PATENT_PARTNER_V2.md', 'data_pipeline/PARTNER_4090_GUIDE.md',
    'data_pipeline/partner_evidence_dataset.py', 'data_pipeline/partner_dense_baseline.py',
    'data_pipeline/partner_evidence_max_baseline.py',
    'data_pipeline/partner_evidence_teacher.py', 'data_pipeline/partner_evidence_teacher_model.py',
    'data_pipeline/partner_teacher_candidates.py',
    'data_pipeline/partner_teacher_calibration.py',
    'data_pipeline/patent_partner_baseline.py', 'data_pipeline/package_partner_data.py',
    'data_pipeline/cooperation_baseline.py',
    'data_pipeline/stage_partner_code.ps1',
    'tests/test_partner_evidence_dataset.py', 'tests/test_partner_dense_baseline.py',
    'tests/test_partner_evidence_max_baseline.py',
    'tests/test_partner_evidence_teacher.py', 'tests/test_partner_evidence_teacher_model.py',
    'tests/test_partner_teacher_candidates.py',
    'tests/test_partner_teacher_calibration.py',
    'tests/test_patent_partner_baseline.py', 'tests/test_partner_transfer.py'
)
foreach ($relative in $sourceFiles) {
    $candidate = Get-Item -LiteralPath (Join-Path $projectRoot $relative)
    if ($candidate.PSIsContainer -or $candidate.Length -gt 10MB) {
        throw "Expected a small source file: $relative"
    }
}
Write-Output 'Only these source files will be staged; data, weights and transfer packages are excluded:'
$sourceFiles | ForEach-Object { Write-Output ('  ' + $_) }
if (-not $Stage) {
    Write-Output 'Run again with -Stage to stage these files. No commit or push is performed by this script.'
    exit 0
}
$existingStaged = @(git -C $projectRoot diff --cached --name-only)
if ($LASTEXITCODE -ne 0) { throw 'Could not inspect the Git index.' }
if ($existingStaged.Count -gt 0) {
    throw 'The Git index already contains staged files. Review it before using this helper; nothing was changed.'
}
& git -C $projectRoot add -- @sourceFiles
if ($LASTEXITCODE -ne 0) { throw 'Git staging failed.' }
& git -C $projectRoot diff --cached --stat
Write-Output 'Review the staged list, then commit and push yourself. This script did not commit or upload anything.'
