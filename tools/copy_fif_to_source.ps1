param(
    [Parameter(Mandatory=$true)][string]$OutputDirectory,
    [Parameter(Mandatory=$true)][string]$InputRoot,
    [int]$MaximumHours=24
)
# 只做派生文件复制，不改权限、不改BDF。必须在允许访问原数据盘的本机PowerShell里运行。
# Python先生成并逐样本核验FIF；这里再核对复制前后的SHA-256。
$ErrorActionPreference='Stop'
$outputFull=[IO.Path]::GetFullPath($OutputDirectory).TrimEnd('\')
$inputFull=[IO.Path]::GetFullPath($InputRoot).TrimEnd('\')
$queue=Join-Path $outputFull 'copy_queue'
New-Item -ItemType Directory -Path $queue -Force | Out-Null
$started=Get-Date
$copied=0
function Write-PrivateJson([string]$Path,$Value){
    $json=$Value | ConvertTo-Json -Depth 12
    [IO.File]::WriteAllText($Path,$json,[Text.UTF8Encoding]::new($false))
}
while(((Get-Date)-$started).TotalHours -lt $MaximumHours){
    $jobs=Get-ChildItem -LiteralPath $queue -Filter '*.ready.json' -File
    foreach($job in $jobs){
        $stem=$job.Name.Substring(0,$job.Name.Length-11)
        $done=Join-Path $queue ($stem+'.done.json')
        $failed=Join-Path $queue ($stem+'.failed.json')
        if((Test-Path -LiteralPath $done) -or (Test-Path -LiteralPath $failed)){continue}
        $created=[Collections.Generic.List[string]]::new()
        try{
            $request=Get-Content -LiteralPath $job.FullName -Raw -Encoding utf8 | ConvertFrom-Json
            $sourceBdf=[IO.Path]::GetFullPath($request.source_bdf)
            if(-not $sourceBdf.StartsWith($inputFull+'\',[StringComparison]::OrdinalIgnoreCase)){throw 'BDF不在本次输入目录内。'}
            if((Get-FileHash -LiteralPath $sourceBdf -Algorithm SHA256).Hash.ToLowerInvariant() -ne $request.source_sha256){throw '复制前源BDF指纹不一致。'}
            $parent=Split-Path -Parent $sourceBdf
            foreach($item in $request.files){
                $from=[IO.Path]::GetFullPath($item.source)
                $to=[IO.Path]::GetFullPath($item.target)
                if(-not $from.StartsWith($outputFull+'\files\',[StringComparison]::OrdinalIgnoreCase)){throw '暂存文件不在本次结果目录内。'}
                if((Split-Path -Parent $to) -ne $parent){throw '派生文件目标与BDF目录不一致。'}
                $name=[IO.Path]::GetFileName($to)
                if($name -notmatch '^fdata(?:_group\d+)?(?:-\d+)?\.fif$' -and $name -ne 'fdata_batch_manifest.json'){throw '只允许发布本轮FIF及转换清单。'}
                $expected=(Get-FileHash -LiteralPath $from -Algorithm SHA256).Hash
                if(Test-Path -LiteralPath $to){
                    if((Get-FileHash -LiteralPath $to -Algorithm SHA256).Hash -ne $expected){throw '目标已有不同内容，拒绝覆盖。'}
                }else{
                    [IO.File]::Copy($from,$to,$false)
                    $created.Add($to)
                }
                if((Get-FileHash -LiteralPath $to -Algorithm SHA256).Hash -ne $expected){throw 'FIF复制后指纹不一致。'}
            }
            Write-PrivateJson $done @{ok=$true;files=$request.files.Count;source_sha256=$request.source_sha256;time=(Get-Date).ToString('o')}
            $copied++
            Write-Output ('Copied and verified jobs: '+$copied)
        }catch{
            foreach($path in $created){Remove-Item -LiteralPath $path -ErrorAction SilentlyContinue}
            Write-PrivateJson $failed @{ok=$false;error=$_.Exception.Message}
            Write-Output 'A copy job failed; details saved privately.'
        }
    }
    if(Test-Path -LiteralPath (Join-Path $outputFull 'files_complete.json')){break}
    Start-Sleep -Milliseconds 300
}
