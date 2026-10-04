# A8 QAT recovery and comparison

## Reference 25,000 evaluated; candidate resumed — October 4, 11:18 UTC

Reference25000 eval exited0: actualsupervisorstart11:01:07.378365Z/end
11:12:31.086497Z,wall683.708132s; reportphase678.563439s,terminalabsence observed
11:12:32Z. Reportd80978e41eb18231f59eb1be1ddb9aa0554e609624802d7d9b4d84c5182ebc60;
checkpointe3948dbd. CE3.550171148/229labels/48loss rounds/all24prompts. A8 accepted
1104/8465proposals over1731rounds,0.637782drafts/round,0.130419acceptance;
Q4_0 remains1608/6047/1231/1.306255. Native emitted2834tokens each.
RequestTPS95.17054 versusQ4_0134.19039 (ratio0.70922),target-only88.41791;
decode99.41128 versus142.78095 (ratio0.69625). Five×24=120requests per variant,
14,290returnedIDs/105length/15stop finishes each;24native/120timing matchesQ4_0,
sealed=false. Reference improves but remains below primaryQ4_0.

Raw attempt/report/result/timing hardlink archived at
`evidence-archive/development/reference-step25000-eval-attempt-0000`, locator
`reference-step25000-locator-map.json`; resultd99f100e...,timingd44833b5...,
checkpointmanifestef8a346e...,jointjson5b792fe9...,jointnpzf43dc3a5.... Eleven
originalreports in compact summary. Exactgroups/session/native contexts returned,
baseline2766MiB/0%. Reference budgetunchanged4324.037866769007/7200,
active_attempt null,remaining2875.962133230993.

Candidate resumed exact20000/cursor20028/checkpoint21f5278f/config5bba/source3bd,
run `a8-candidate-step20000-resume-supervisor-20261004-01`, host session
`a8-candidate-step20000-resume-20261004-01`, actualstart11:13:57.708125Z.
Supervisor25518/birth4699500,trainer25523/birth4699507,sameboot517c4a36,
faulthandler diagnostic only. Positive resume observed11:18:03Z: step20637/
cursor20668,training4280.945520s,finite33gradients,cumulative sign flips649521778,
expectedserial learned-head path. Active original reservation3067.803136 from
base4132.196864 retained. Continue SAME handles to natural25000, normal1200s
native eval/archive/release, then reference25000→30000. No repeated gates/science.

Both7200 endpoints/finals incomplete; Goal/heartbeat ACTIVE, allsource/recipe/
math/data/precision/budget invariant. SIGSEGV retry1max2/charges retained,
A1 held out, no2080host controls. Currentcanonicalobservations/exactidentities
and actual versus observation timestamps govern; no restart from stale snapshots.

## Reference 25,000 checkpoint; evaluation running — October 4, 11:01 UTC

Reference reached natural25000/cursor25044, boundary observed10:59:58Z.
Checkpoint SHA
`e3948dbd55be4b443d1d0ff8bb2c6e4b5c08b3e8308e69c435b8692b5c3088a8`
hardlink archived with manifestef8a346e...,status/budget/supervisor evidence;
locator `evidence-archive/checkpoints/reference-step-25000-locator.json`.
Supervisor actualstart10:44:41.639338Z/end10:59:26.369718Z,exit0. Settled
budget4324.037866769007/7200,active_attempt null,remaining2875.962133230993.
Exact24320/4523893 and24325/4523900 groups/session/server absent. Training resource baseline
2766MiB/0% before scheduled evaluation.

Native evaluation launched11:01:06Z, same immutable3bd/current9e2/F16 target/
verifier/KV/frozen24dev/1200s cap, diagnostic faulthandler only. Host tmux session
`a8-reference-step25000-eval-20261004-01`, run
`a8-reference-step25000-eval-supervisor-20261004-01`; supervisor24696/birth4622464,
evaluator24701/birth4622474 on sameboot517c4a36. Existing25k manifest SHAfc18f400.
Initial11:01:17 resource snapshot startup2766MiB/0%; result pending. Earlier
resource timestamp10:00:16 is retracted: raw04:00:16PDT corresponds to11:00:16UTC.
Use explicit UTC timestamps; no resource/recipe/budget change resulted.

Candidate remains evaluated20000/4132.196864,0.205183drafts/round/67.751requestTPS,
below ref20000 andQ4_0. After ref terminal archive/release, resume candidate
20000→25000 under3067.803136remaining, normal evaluation, then reference25000→30000.
Both7200 cumulative endpoints/final evaluations remain incomplete; Goal/heartbeat
ACTIVE, allsource/recipe/math/data/precision/budgets unchanged, no repeated gates.
SIGSEGV retry1/max2/charges retained, A1 held out, no2080host controls. Current
canonical operator observations/births/checkpoint IDs govern; separate actual
start/end from later observations and do not restart from stale legacy fields.

## Candidate 20,000 evaluated; reference resumed — October 4, 10:47 UTC

Candidate20000 eval exited0: supervisor actualstart10:28:34.371620Z/end
10:41:31.094831Z,wall776.723211s; reportphase771.596616s,terminalabsence observed
10:42:14Z. Report624c8e3397bd5c5c6a6deb6e88431e268ba9af80bcdaaa5f4412767b220a3d64;
checkpoint21f5278f. CE5.781816153/229labels/48loss rounds/all24prompts. A8 accepted
483/11522proposals over2354rounds,0.205183drafts/round,0.041920acceptance;
Q4_0 remains1608/6047/1231/1.306255. Native emitted2834tokens each.
RequestTPS67.75124 versusQ4_0135.81222 (ratio0.49886),target-only88.85102;
decode69.61893 versus143.27966 (ratio0.48590). Five×24=120requests per variant,
14,290returnedIDs/105length/15stop finishes each;24native/120timing matchesQ4_0,
sealed=false. Candidate still well below ref20000 andQ4_0.

Raw attempt/report/result/timing hardlink archived under
`evidence-archive/development/candidate-step20000-eval-attempt-0000`, locator
`candidate-step20000-locator-map.json`; resultc2c9280a...,timinga505b986...,
checkpointmanifest0521d072...,jointjsonb5820510...,jointnpz2d714500.... Ten original
reports in compact summary. Exact groups/session/native contexts returned,
baseline2766MiB/0%. Candidate budget unchanged4132.196864/7200,active_attempt null,
remaining3067.803136. An operator deadline note incorrectly added40minutes;
corrected nominal10:48:33 (not11:08:33), actual source/manifest cap stayed1200s.
No timeout expansion occurred.

Reference resumed exact20000/cursor20028/checkpoint1fab73c3/configcf320/source3bd,
run `a8-reference-step20000-resume-supervisor-20261004-01`, host session
`a8-reference-step20000-resume-20261004-01`, launch10:44:40Z.
Supervisor24320/birth4523893,trainer24325/birth4523900,sameboot517c4a36,
faulthandler diagnostic only. Actual positive resume observed10:47:21Z: step20302/
cursor20331,training3603.327545s,finite18gradients,cumulative sign flips32505352,
effective batched cache/chunk64. Active original reservation3658.642051 from base
3541.357949 retained. Continue SAME handles to natural25000, then1200s normal
native eval/archive/release and candidate20000→25000. No repeated gates or science.

Both7200 endpoints/finals still incomplete; Goal/heartbeat ACTIVE, all admitted
source/recipe/math/data/precision/budget invariant. SIGSEGV retry1/max2/charges
retained, A1 held out, no2080host controls. Current canonical operator observations
and exactbirths govern, no restart from stalelegacy fields or delivery order.

## Candidate 20,000 checkpoint; evaluation running — October 4, 10:28 UTC

Candidate training supervisor started10:08:25.718304Z and exited0 at
10:27:26.033232Z; boundary observed10:27:27Z. Natural step20000/cursor20028,
checkpoint SHA
`21f5278f9489b166067a59d83f22f454cacca975e419b0a010a15cdf7df75865`
hardlink archived; locator `evidence-archive/checkpoints/candidate-step-20000-locator.json`.
Checkpoint manifest0521d0721b93c3b8fa3c5c15dac44ade5be62e7de010de6f83805e4a913437bc;
jointjsonb5820510...,jointnpz2d714500..., exact paths/full hashes in operator record.
Settled budget4132.196864/7200,active_attempt null,remaining3067.803136. Exact
22738/22743 groups/session/contexts returned,baseline2766MiB/0%.

Scheduled evaluation launched10:28:33Z, same immutable3bd/current9e2/F16 target/
verifier/KV/frozen24dev/1200s cap, diagnostic faulthandler only. Host tmux session
`a8-candidate-step20000-eval-20261004-01`, run
`a8-candidate-step20000-eval-supervisor-20261004-01`; supervisor23230/birth4427166,
evaluator23235/birth4427174 on sameboot517c4a36. Exact existing manifest SHAfc18f400.
Initial10:28:50 resource snapshot startup2766MiB/0%; result pending.

Reference stays evaluated20000/3541.357949,0.613546drafts/round/93.850requestTPS,
belowQ4_0. After candidate terminal archive/release, resume reference20000→25000
under3658.642051 remaining, then normal evaluation and candidate20000→25000.
Both7200 endpoints/finals still incomplete; Goal/heartbeat ACTIVE, all admitted
science/source/recipe/data/precision/budget invariant, no repeated gates. SIGSEGV
retry1/max2/charges retained, A1 held out, no2080host controls. Canonical current
operator state latest_operator_observation; exact end versus observation distinct.

## Reference 20,000 evaluated; candidate resumed — October 4, 10:10 UTC

Reference20000 native evaluation completed exit0: supervisor actualstart
09:54:36.659778Z/end10:06:03.350638Z,wall686.690860s; reportphase681.765045s,
terminalabsence observed10:06:45Z. Report
3e4d03de6864109075dfb356dfa66e78ec9f326fb461e69ae53f3b8524c20721,
checkpoint1fab73c3. CE3.699926959/229labels/48loss rounds/all24prompts. A8 accepted
1078/8590proposals over1757rounds,0.613546drafts/round,0.125495acceptance;
Q4_0 remains1608/6047/1231/1.306255. Native emitted2834tokens each.
RequestTPS93.84966 versusQ4_0134.04624 (ratio0.70013), target-only88.44147;
decode97.89851 versus142.26941 (ratio0.68812). Five×24=120requests per variant,
14,290returnedIDs/105length/15stop finishes each;24native/120timing matchesQ4_0,
sealed=false. Reference improves but remains below primaryQ4_0.

Raw attempt/report/result/timing hardlink archived at
`evidence-archive/development/reference-step20000-eval-attempt-0000`, locator
`reference-step20000-locator-map.json`; result57a60fb7...,timingd9291b99...,
checkpointmanifest35bbe4b8...,jointjson2190be07...,jointnpzc115430c....
Nine original complete reports in compact summary; exact session/groups/native
contexts returned, baseline2766MiB/0%. Reference budget unchanged3541.357949,
active_attempt null,remaining3658.642051.

Candidate resumed exact15000/cursor15024/checkpoint50e0764c/config5bba/source3bd,
run `a8-candidate-step15000-resume-supervisor-20261004-01`, host session
`a8-candidate-step15000-resume-20261004-01`, actualstart10:08:25.718304Z.
Supervisor22738/birth4306301,trainer22743/birth4306308,sameboot517c4a36,
faulthandler diagnostic only. Positive resume observed10:10:34Z: step15091/
cursor15115,training3124.298694s,finite33gradients,cumulative sign flips443023080,
expected serial learned-activation head path. Active reservation4106.879955265
from base3093.120044735 retained. Continue SAME handles to natural20000,
normal1200s evaluation/archive/release, then reference20000→25000.

Both7200 cumulative endpoints/finals remain incomplete; Goal/heartbeat ACTIVE.
No repeated admissions or source/recipe/math/data/precision/budget changes,
SIGSEGV retry1/max2/charges retained, A1 held out, no2080host controls.
Use canonical latest_operator_observation/currentest observed_at and actual
start/end versus later observation time; no restart from stale legacy fields.

## Reference 20,000 checkpoint; evaluation running — October 4, 09:54 UTC

Reference training supervisor started09:37:30.071130Z and exited0 at
09:52:02.807364Z; boundary observed09:52:11Z. Natural step20000/cursor20028,
checkpoint SHA
`1fab73c38bab8992fb0c92a0a56691472674ec56a0d465f1fe7b6efa5169ab8d`
hardlink archived, locator `evidence-archive/checkpoints/reference-step-20000-locator.json`.
Checkpoint manifest35bbe4b8...,A8/joint.json2190be07...,A8/joint.npzc115430c...;
exact full hashes in operator evidence. Settled budget3541.357949/7200, active
attempt null,remaining3658.642051. Exact21309/21314 groups/session/contexts
returned, baseline2766MiB/0%. Actual end/observation timestamps remain separate.

The existing scheduled-manifest procedure materialized the missing20k manifest
with the SAME frozen fc18f400 hash; no new gate or data. Native evaluation
launched09:54:35Z under same immutable3bd/current9e2, fixed24dev/F16 target/verifier/
KV/1200s cap, diagnostic faulthandler only. Run
`a8-reference-step20000-eval-supervisor-20261004-01`, host tmux session
`a8-reference-step20000-eval-20261004-01`; supervisor21840/birth4223396 and
child21845/birth4223402 on boot517c4a36. Initial09:54:51 resource snapshot was
startup at2766MiB/0%; result pending. Current operator state is canonical
`latest_operator_observation`, not stale legacy training phase fields.

Candidate remains evaluated15000/3093.120044735,0.209471drafts/round/68.261requestTPS,
below ref15000 andQ4_0. After reference terminal archive/release, resume candidate
15000→20000 under4106.879955265 remaining. Both7200 endpoints/finals remain
incomplete; Goal/heartbeat ACTIVE, recipes/math/data/precision/budgets unchanged,
no repeated admissions. SIGSEGV retry1/max2/charges retained, A1 held out,
no2080host controls or unrelated research.

## Candidate 15,000 evaluated; reference resumed — October 4, 09:49 UTC

Candidate15000 evaluation completed exit0: supervisor started09:22:04.135414Z,
ended09:34:57.839048Z, wall773.703634seconds. Report evaluation phase769.606530s;
terminal absence/GPU observed09:35:23Z. Preserve these distinct clocks.
Report445e3af8925102393dc4ac0cd86d93463d69010a5f2d2e76eb2ee9a53e9a19b9;
checkpoint50e0764c. CE5.951622479/229labels/48loss rounds/all24prompts. A8 accepted
491/11457proposals over2344rounds,0.209471drafts/round,0.042856acceptance.
RequestTPS68.26114 versusQ4_0136.43652 (ratio0.50031), target-only89.32846;
decode69.89315 versus143.17237 (ratio0.48817). Five×24=120requests per variant,
14,290returned IDs/105length/15stop finishes each;24native/120timing matchesQ4_0,
sealed=false. Slight recovery from10000 but still below reference15000 andQ4_0.

Raw attempt/report/result/timing hardlink archived at
`evidence-archive/development/candidate-step15000-eval-attempt-0000`, locator
`candidate-step15000-locator-map.json`. Result27635216...,timing1667c7dc...;
eight original complete reports in compact summary. Exact eval/session/groups/
contexts returned, baseline2766MiB/0%. Candidate budget remains3093.120044735,
active_attempt null,remaining4106.879955265. Verified checkpoint `manifest.json`
SHA bdef1786b0f869542588c02d55105746afde3a2efb7c82c394367afaabb4e7b8;
A8/joint.json5f1f5c8aa58f2250807e7cdcce732046b6b544f279e1e32ba48401665d062964,
A8/joint.npz994c0e1fdf76d1da5ec1f131a83da9472330dcb82cc3f96153b092910ac238c0.
Earlier1e8c9381 abbreviated 'manifest' is retracted as unlocated/mislabel, not
an identified alternate artifact. Actual matching resume/archive hashes are stable.

Reference resumed exact15000/checkpoint73429bd0/configcf320476/source3bd,
run `a8-reference-step15000-resume-supervisor-20261004-01`, host session
`a8-reference-step15000-resume-20261004-01`, started09:37:29Z.
Supervisor21309/birth4120738,trainer21314/birth4120744,sameboot517c4a36,
faulthandler diagnostic only. Positive resume15081/cursor15105 at09:39:30Z;
latest09:48:40 step18687/cursor18714,training3342.323s,finite18gradients,
cumulative sign flips27780465, effective batched cache/chunk64. Active original
reservation4432.005199101 from base2767.994800899 preserved. Continue SAME handles
to natural20000 evaluation, then candidate15000→20000. No repeated gates or science.

Canonical current operator state is top-level `latest_operator_observation`;
legacy `training.status`/`last_remote_status_observation` can be historical. Root
had read the legacy fields and left its own registration older; current packet/
births/checkpoint resolve that. Re-read and choose newest authoritative observed_at.
Goal/heartbeat ACTIVE; both7200 endpoints/finals incomplete. SIGSEGV retry1/max2
and charges preserved, A1 held out, no2080host control or unrelated research.

## Candidate 15,000 checkpoint; evaluation running — October 4, 09:22 UTC

Candidate training exited0 at09:19:26.091523Z, natural step15000/cursor15024.
Checkpoint SHA
`50e0764ce0a7c343366c04827605b6c12b5fc82f87dec5063ccea316bf12f102`
hardlink archived at `evidence-archive/checkpoints/candidate-step-15000`, with
status/budget/trainer evidence. Settled budget3093.120044735/7200 seconds,
active_attempt null,remaining4106.879955265. Exact trainer19759/supervisor19753
are absent; resource baseline2766MiB/0%. Actual end differs from later observation
09:19:45; retain both timestamps separately.

Scheduled evaluation launched09:22:04.135414Z under same immutable3bd/current9e2,
exact frozen24dev requests/F16 target/verifier/KV/1200s cap, diagnostic faulthandler
only. Run `a8-candidate-step15000-eval-supervisor-20261004-01`, host tmux session
`a8-candidate-step15000-eval-20261004-01`; supervisor20208/birth4028141 and
evaluator20214/birth4028150, sameboot517c4a36. Result pending. Operator will record
manifest hashes with exact paths to disambiguate two abbreviated 'manifest' values;
that metadata clarification is pending. Resume50e0764c is consistent.

Reference remains evaluated15000/2767.994800899,0.581382drafts/round/92.340requestTPS,
belowQ4_0 and above target-only pooled request rate in this measurement. After
candidate terminal archive/resource release, resume reference15000→20000 under
remaining4432.005199101, then its normal1200s evaluation and candidate15000→20000.

Both7200 cumulative endpoints/finals remain incomplete; Goal/heartbeat ACTIVE.
No repeated admissions, source/recipe/math/data/precision/budget changes or research
pivot; SIGSEGV retry1/max2 and charges retained, A1 held out, no2080host controls.

## Reference 15,000 evaluated; candidate resumed — October 4, 09:02 UTC

Reference15000 native evaluation completed exit0 at08:58:33.815436Z,683.691812
supervised seconds. Reportcc5341b12d5733465f5b4ceecc10b9b8c0671ab33a39435703c1cfe23057af74;
checkpoint73429bd0. CE3.994632131/229labels/48loss rounds/all24prompts. A8 accepted
1043/8769proposals over1794rounds,0.581382drafts/round,0.118942acceptance;
Q4_0 remains1608/6047/1231/1.306255. Native emitted2834tokens each.
RequestTPS92.34039 versusQ4_0134.68578 (ratio0.68560), target-only88.55608;
decode96.16208 versus142.76261 (ratio0.67358). Five×24=120requests per variant,
14,290returnedIDs/105length/15stop finishes each;24native/120timing sequences
matchQ4_0, sealed=false. Reference improves and exceeds target-only pooled request
rate in this measurement, but remains belowQ4_0. No significance/global-win claim.

Raw attempt/report/result/timing hardlink archived at
`evidence-archive/development/reference-step15000-eval-attempt-0000`; locator
`reference-step15000-locator-map.json`. Result8490312b...,timing0c1ade69...;
compact summary now contains seven original complete reports. Exact evaluation/
native groups and contexts returned, baseline2766MiB/0%. Reference budget remains
2767.994800899/7200,active_attempt null,remaining4432.005199101.

Candidate resumed exact10000/cursor10017/checkpoint6898e9b1/config5bba2e/source3bd,
run `a8-candidate-train-after-reference15000-supervisor-20261004-01`, host session
`a8-candidate-train-after-reference15000-20261004-01`,start09:00:37.290456Z.
Supervisor19753/birth3899454,trainer19759/birth3899465,sameboot517c4a36;
faulthandler diagnostic only. Actual positive resume at09:02:25Z: step10015/
cursor10032,15updates,finite33gradients,cumulative sign flips254073145. Active
reservation5135.739885756 from prior2064.260114244 retained. Continue SAME handles
to natural15000, scheduled1200s eval/archive/release, then reference15000→20000.

Both7200 endpoints/final evaluations remain incomplete; Goal/heartbeat ACTIVE.
No repeated gates or new science/recipe/data/precision/budget change. Equivalent
unexplained SIGSEGV retry1/max2 retained; A1 held out, no2080host controls.

## Reference 15,000 checkpoint; scheduled evaluation running — October 4, 08:47 UTC

Reference training exited0 at08:43:32Z, natural step15000/cursor15024. Checkpoint
SHA `73429bd0ca2302147a2ee7cf883664a63323da5e229f0e3611a6f5e044a940eb`
hardlink archived with manifest6e88e2f0...80712, status/budget/supervisor evidence.
Settled trainer budget2767.994800899/7200, active_attempt null,
remaining4432.005199101. Exact supervisor18551/trainer18557 groups and contexts
returned; baseline RTX50802766MiB/0%. Same source3bd/configcf320476 authenticated.

Scheduled native evaluation launched08:47:10.123624Z, current9e2, faulthandler
only, same frozen24 prompts/F16 target/verifier/KV/1200s cap. Run
`a8-ref-step15000-eval-supervisor-20261004-01`, host tmux session
`a8-ref-step15000-eval-20261004-01`; supervisor18949/birth3818741 and evaluator
18955/birth3818749, boot517c4a36. Result pending. Candidate remains evaluated10000/
2064.260114244,0.180342drafts/round/66.145requestTPS. After ref terminal archive/
release, resume candidate10000→15000 under remaining5135.739885756.

Older08:29/08:30 messages arrived after the15k boundary, and ledger phase/timestamps
regressed while a newer nested observation remained. Root requested a fresh compact
packet rather than restarting. Operator confirmed actual terminal08:43:32 separately
from observation08:46 and corrected current phase; same15k SHA retained. See
USER_LESSONS. Preserve current observed_at and actual end times separately.

Goal/heartbeat ACTIVE. Both7200 endpoints/final evaluations remain incomplete;
no repeated admissions, source/recipe/data/precision/budget changes or research
pivot. Equivalent unexplained SIGSEGV retry1/max2 policy retained; A1 held out,
no RTX2080Ti controls. Latest operator ledger owns current command/phase evidence.

## Candidate 10,000 evaluated; reference resumed — October 4, 08:30 UTC

Candidate10000 evaluation completed exit0 at08:24:42.063721Z,783.215612 supervised
seconds. Reporte561e03ce99c049458d815239f7c0d8ea3433abcaf5be63cbc31ad032fa5acbd;
checkpoint6898e9b1. CE6.028492996/229labels/48loss rounds/all24prompts. A8 accepted
433/11724 proposals over2401rounds,0.1803415drafts/round,0.0369328acceptance;
Q4_0 remains1608/6047/1231,1.306255drafts/round. Both emitted2834tokens.
RequestTPS66.14480 versus135.03076 (ratio0.48985), target-only88.59359;
decode68.04333 versus142.87483 (ratio0.47624). Five×24=120requests per variant,
14,290returned IDs and105length/15stop finishes each;24native/120timing sequences
matchQ4_0, sealed=false. Candidate regressed versus5000; both10k arms trailQ4.

Raw attempt/report/result/timing hardlink archived at
`evidence-archive/development/candidate-step10000-eval-attempt-0000`, locator
`candidate-step10000-locator-map.json`. Result3bbbb913...,timingfc12102...;
compact summary contains six complete original reports. Exact eval groups/native
contexts returned, baseline2766MiB/0%. Candidate budget unchanged2064.260114244,
active_attempt null; reference stays2005.419445274, inactive. Continue unchanged
science: next reference10000→15000 under remaining5194.580554726, then its1200s
eval/archive/release and candidate10000→15000 (remaining5135.739885756).

Reference resumed exactly from10000/SHA281546e7/configcf320476/source3bd,
run `a8-ref-train-after-candidate10000-supervisor-20261004-01`, host tmux session
`a8-ref-train-after-candidate10000-20261004-01`, started08:28:58.538173Z.
Supervisor18551/birth3709579, trainer18557/birth3709590, sameboot517c4a36,
faulthandler diagnostic only. Positive resume at08:30:52Z: step10058/cursor10075,
58updates, finite18gradients, cumulative sign flips8973372. Active reservation
5194.580554726 from settled base2005.419445274 retained. Continue SAME handles
toward scheduled15000 evaluation; no new admission or science changes.
No recipe/data/precision/budget changes or added diagnostics solely from the
negative candidate result. Goal/heartbeat ACTIVE, both7200 endpoints/finals still
incomplete; unexplained SIGSEGV retry1/max2 retained, A1 held out, no2080controls.

## Candidate 10,000 checkpoint; scheduled evaluation running — October 4, 08:11 UTC

Candidate reached natural step10000/cursor10017 and status awaiting_development
at08:10:33Z. Checkpoint SHA
`6898e9b1cb6cb0303e1a38340a23decb3eeabf14d215afd761d7e0c8aa7496ae` is preserved
in verified hardlink archive `evidence-archive/checkpoints/candidate-step-10000`.
Budget settled2064.260114244/7200 seconds, active_attempt null,
remaining5135.739885756; prior16980/16986 training leaders absent.

Scheduled native evaluation launched08:11:38.848109Z under same immutable3bd,
current9e2 runtime, exact frozen24dev requests, F16 target/verifier/KV and existing
1200s cap. Run `a8-candidate-step10000-eval-supervisor-20261004-01`, host tmux
session `a8-candidate-step10000-eval-20261004-01`; supervisor17438/birth3605614,
evaluator17444/birth3605621 on same boot517c4a36. Diagnostic fault handler only;
no new gates, recipe/math/data/precision/budget change. Result pending.

Reference remains evaluated10000/2005.419445274 with request88.069TPS and
0.500529accepted drafts/round, below Q4_0. Both are now at10000 committed updates;
accounted time differs (candidate2064.260, reference2005.419 including failed work
and conservative downtime). Comparison still requires BOTH7200 endpoints/final
native evaluations. After candidate terminal archive/resource release, resume
reference10000→15000. Same Goal/heartbeat ACTIVE; unexplained SIGSEGV retry1/max2
policy preserved, A1 held out, no2080host-control writes or unrelated science.

## Reference 10,000 evaluated; candidate resumed — October 4, 07:53 UTC

Reference10000 evaluation completed exit0 at07:47:30.913917Z, 692.18257 supervised
seconds. Report3481aeda0ac35e28a6579c3fee9b25ef8e554e459504c5a3212abaa039de2d7c;
checkpoint281546e7. CE4.262763084 on229labels/48loss rounds/all24prompts. A8 accepted
946/9238 proposals over1890rounds, 0.500529drafts/round versus Q4_0 at1.306255;
both emitted2834tokens. Request TPS88.06884 versus135.09797 (ratio0.65189) and
target-only88.78173; decode91.44398 versus142.97351 (ratio0.63959). Five×24=120
measured requests per variant,14,290returned tokenIDs each; all24 native and120
request sequences match Q4_0. No sealed access. Reference improves but is below Q4.
Raw report/result/timing/attempt archived under `evidence-archive/development/reference-step10000-eval-attempt-0000`;
locator `reference-step10000-locator-map.json`. All exact evaluation/native groups
and contexts returned; baseline2766MiB/0%. Reference budget remains2005.419445274,
active_attempt null; SIGSEGV retry1/max2 remains operationally recovered/unexplained.

Candidate resumed from archived5000/cursor5009/SHA893e205d under same immutable3bd,
config5bba2e/source/recipe, diagnostic faulthandler only. Supervisor run
`a8-candidate-train-after-reference10000-supervisor-20261004-01`, session
`a8-candidate-train-after-reference10000-20261004-01`, start07:51:37.693156Z.
Supervisor16980/birth3485495, trainer16986/birth3485506, same boot517c4a36.
Positive optimizer resume verified at07:53:48Z: step5133/cursor5142,133updates,
finite33gradients, cumulative sign flips80762347, scale L1 movement0.16244.
Budget prior1047.284754942/7200 and active reservation6152.715245058 preserved.
Continue SAME handles to natural10000 scheduled evaluation, then reference10000→15000.
No new gates, recipe/data/precision/budget changes or third retry; Goal/monitor ACTIVE.
Both cumulative endpoints and final reports remain incomplete.

## Reference retry 1 reached 10,000; evaluation running — October 4, 07:35 UTC

Reference supervisor15464/trainer15470 exited −11 at 07:24:59.912725Z. Last status
step9681/cursor9697, cumulative observed trainer time1470.676493 seconds. Latest
durable checkpoint is step9000/cursor9016, resume SHA
`4aba3effdf2d9bae5bd80b5dd1f7ae027356c4b4c9bbc63b11fc2c56e0714e0d`, manifest
`3927be237de75978134f13564ce0c14de8caa3ac2d092355cea1b316add426bc`, joint NPZ
`7f1860d5a9b0959bb4e78e4dd92f67d4737f5129b2dd0ea4b32acb1551529bb5`.
Failure state/stdout/status/budget/latest snapshots are preserved under
`evidence-archive/failures/reference-training-segfault-attempt-001`; checkpoint
hardlink archive `evidence-archive/checkpoints/reference-crash-step-9000` verified.
No Python traceback or core appeared. All exact failed groups/children/native
contexts/session returned; no compute apps, baseline RTX5080 2766 MiB/0%.

Same-boot built-in TrainingBudget reconciliation charged1085.811027178 seconds to
the failed attempt, retaining all failed post-checkpoint work. Settled budget now
1850.536234645/7200 seconds; remaining5349.463765355. This includes approximately
379.457 seconds of conservative downtime beyond known terminal time (derived from
recorded start/terminal clocks); no refund or budget extension. Committed coverage
rolls back to checkpoint9000, while failed elapsed time remains charged.

Retry1 launched 07:29:55.913250Z under SAME immutable3bd/configcf320476/boot517c4a36.
Supervisor15845/birth3355318 and trainer15851/birth3355328; diagnostic environment
`PYTHONFAULTHANDLER=1` only. Actual positive resume reached step9241/cursor9257 at
07:32:14Z, 241 updates, finite18 gradients. New active reservation5349.463765355,
start07:31:32.012Z; optimizer/RNG/cursor/recipe checkpoint restored. Retry1 then
passed failed9681/cursor9697 and reached step10000/cursor10017 at
07:34:09.987710Z, exit0/status awaiting_development. Checkpoint SHA
`281546e75c0d1d010a32b5477890788ad77077e1ec6819ceae745a2ac5f24f78`.
Budget settled2005.419445274/7200, active_attempt null, remaining5194.580554726;
retry itself charged154.883210629seconds. Last update finite18 tensors, cumulative
sign flips8914695. Groups/context released, baseline2766MiB/0%. Post-recovery
budget snapshot hardlink preserved (SHA5dea423c...14d43).

Reference10000 scheduled evaluation launched07:35:58.731347Z, same3bd/native9e2,
run `a8-ref-step10000-eval-supervisor-20261004-01`, session
`a8-ref-step10000-eval-20261004-01`; supervisor16090/birth3391596 and evaluator16096/birth3391610.
Operator ledger preserves exact identities. Checkpoint hardlink archive `evidence-archive/checkpoints/reference-step-10000`
verified, manifest05703693...204ca. Evaluation cap1200 unchanged; result pending.
Then candidate resumes from evaluated5000/1047.285s toward10000.

Incident is operationally recovered, cause unexplained. Equivalent future training
SIGSEGV remains the same incident unless a distinct cause is proved;
retry_count1/max2 AUTOMATIC RETRIES after original failure. No retry3, new
science/recipe/precision/data, feature disabling or repeated admission. Goal and
heartbeat ACTIVE; sole operator ledger owns fresh state. Both endpoints incomplete.

## Candidate 5,000 evaluated; reference positive resume — October 4, 07:14 UTC

Candidate evaluation completed exit 0 at 07:06:52.661423Z; supervised runtime
761.22327 seconds. Report SHA b5845c6340d8a70fb5f3661d48ee469ad54a7ba69d1039b563dab81882ce54f9;
checkpoint 893e205d. Accepted 507/11,371 proposals over 2,327 rounds:
0.217877 accepted drafts/round, versus Q4_0 at 1.306255. CE 5.568934 on 229 labels,
48 loss rounds and all 24 prompts. Request rate 68.34155 versus Q4_0 at 135.15673
and target-only at 88.64152 tokens/s; Q4 ratio 0.50565. Decode rate 70.43144 versus
143.25322, ratio 0.49166. All 24 native output sequences and 120 timing sequences
match Q4_0, including target-only timing. Five repetitions × 24 prompts = 120
measured requests per variant. No sealed access. Candidate trails reference at
this update count; final equal-time endpoints remain pending. The combined
candidate does not isolate individual feature effects.

Raw attempt/report/result/manifest/timing and checkpoint archives are hardlink
verified. Locator `evidence-archive/development/candidate-step5000-locator-map.json`.
All evaluation/native leaders and groups are absent; remote session terminal,
no compute apps, baseline RTX5080 2,766 MiB/0%. Candidate budget unchanged at
1,047.284754942/7,200 seconds with no active training attempt.

Reference resumed from its evaluated checkpoint 884dd8c5/cursor5009 under the
same immutable 3bd source/config. Supervisor run
`a8-ref-train-after-candidate5000-supervisor-20261004-01`, host tmux session
`a8-ref-train-after-candidate5000-20261004-01`, started 07:11:36.703841Z.
Supervisor15464/birth3245398 and trainer15470/birth3245407 on boot517c4a36.
Positive resumed optimizer work is verified: step5411/cursor5421 at 07:14:18Z,
411 updates past the exact checkpoint. Cumulative budget829.815687817/7200 seconds;
active attempt retains prior764.725207467 and reservation6435.274792533,
PID15470/birth3245407. All18 gradient tensors finite, sign flips1795 and scale
L1 movement0.135362. Continue the same handles toward scheduled10,000 evaluation.
SAME Goal/heartbeat active, no recipe/data/precision/budget change.

The clean, fully merged helper-only repair worktree and branch were retired;
only disposable caches were ignored. Immutable remote 3bd execution and all raw
runs remain preserved. Current counts and exact identities are in the ignored
operator ledger and compact comparison summary.

## Endpoint evidence audit — October 4, 07:20 UTC

Trainer owner inspected current source66b0ea1 only: no edits, tests, remote actions
or new gate. Final committed coverage is in authenticated `resume.pt` sets
`unique_prompts`/`unique_rows`, plus tokens/epoch/cursor; status exposes counts as
`unique_prompts`, `unique_supervised_rows`, `presented_supervised_tokens`. Restores
preserve sets/cursor; replay increases presented tokens without inflating distinct
coverage. Failed rolled-back work can remain budget-charged outside committed sets.

`status.models.A8.sign_flips` is ONE optimizer update; `cumulative_sign_flips` is
exact per-update history. `recipe-audit.json.telemetry` sign-flip/flip-back totals
are sampled diagnostic history: record observation step/gap and avoid conflation.
It also holds near-zero distances/fraction and per-family L1 movement. Aggregate
near-zero threshold0.01 differs from layer threshold0.05. Last sample can precede
endpoint by99steps; movement admission samples one maximum-gradient element per
tensor. Root report now documents these limits; sole operator has extraction paths.

Endpoint evidence must join settled `budget-used.json` training_seconds7200 and
active_attempt null to latest final checkpoint, `development-request.json`
final_training_complete, completed `development-result.json`, matching report
checkpoint/split/runtime/complete timing and terminal status. In-flight work can
finish after the capped deadline; null budget does not prove physical release.
Both cumulative endpoints and final evaluations remain INCOMPLETE. Continue the
same admitted comparison without repeated proof runs or added science.

## Archived primary acceptance counts transcribed — October 4, 07:04 UTC

The sole operator saved compact original report summaries locally at ignored
`runs/qat-a8-recovery/comparison-interim-summary.json`, with joined report/result/
timing/checkpoint hashes and immutable native runtime identities. Reference and
candidate step-zero both accepted309of12354proposals across2525rounds:
0.122376accepted drafts/round. Reference5000 accepted777of10077 across2060rounds:
0.377184drafts/round. Q4_0 accepted1608of6047 across1231rounds:1.306255drafts/round.
All native captures emitted2834tokens; A8/Q4 response IDs match all24prompts.
Each loss pass used24prompts/48rounds/229labels. Timing is a separate five-repeat,
120-request-per-variant workload. Report now distinguishes both denominators and
pooled latency distributions. Reference remains below Q4_0 and target-only speed;
trained candidate result and cumulative cap endpoints remain pending.

## Scheduled candidate evaluation running — October 4, 06:54 UTC

The sole operator dispatched candidate5000 evaluation under detached host tmux
socket `binary-eagle-runtime`, session `a8-candidate-step5000-eval-20261004-02`.
Supervisor run `a8-candidate-step5000-eval-supervisor-20261004-02` started
06:54:11.438153Z, state running, supervisor PID/PGID14249 and child14255.
The evaluator was in CUDA startup at first snapshot; no completed result yet.
Immutable3bd checkout is clean. Candidate5000 checkpoint hardlink archive is
verified at `evidence-archive/checkpoints/candidate-step-5000`, SHA893e205d.
Actual boot/birth/command/runtime/derived manifest and archive locator are tracked
by the sole operator ledger. Same1200s evaluation cap and frozen24 requests.
After terminal evaluation/archive/resource release, reference resumes5000→10000.

## Candidate reached scheduled 5,000 boundary — October 4, 06:52 UTC

Fresh sole-operator observation confirms candidate step5000/cursor5009,
status awaiting_development. Checkpoint SHA
`893e205d4e64fa04457c3281980570bb5ad13b3fb803cc1fe5f30d2d8b51ad10`;
settled cumulative trainer budget1047.284754942of7200seconds, active_attempt null.
The resumed trainer13691/birth2974154 and supervisor13685/birth2974143 are absent.
RTX5080 observed2766MiB/0%; this residual host memory is not an active A8 trainer.
The existing scheduled candidate5000 native development evaluation is next under
the same1200s cap/current9e2 runtime; its result is pending. Then resume reference
from evaluated5000/764.725s toward its next natural10000 boundary. Recipes, data,
precision and independent budgets remain unchanged; A1 held out, no2080controls.

The tmux MCP client's WSL known-host key is now pinned from the exact existing
Mac trusted public key with strict checking; same WSLboot517c4a36 authenticated.
No host-key rotation or job failure occurred. Last stale3071 evidence is superseded.
SAME Goal/heartbeat ACTIVE; sole operator owns launch/terminal/archive evidence.

## SSH observation recovery — October 4, 06:51 UTC

The presented RTX5080 ED25519 fingerprint matches the Mac's existing trusted
known-host key exactly. The operator's prior alleged mismatch was a hashed
host-label salt mistaken for a key fingerprint; the operator acknowledged it.
The same authenticated public key can be pinned in the actual tmux MCP client's
known-host file with strict checking. No human confirmation or host change is
needed. Sole operator is reconnecting to observe the original process handles;
last confirmed candidate step3071/cursor3076 remains stale. Do not infer failure
or restart from transport silence. Continue the existing candidate5000 scheduled
evaluation and natural-boundary alternation after fresh evidence.

## Actual human-pause resume verified — October 3, 23:28 PDT

Candidate has restored step2081/cursor2084 and advanced to2221/cursor2224 with
REAL optimizer updates. Same clean immutable source3bd/config/optimizer/RNG/
cursor/recipe; fresh boot517c4a36 availability and archived checkpoint SHA matched.
Supervisor13685/birth2974143 and trainer13691/birth2974154 are live. All33gradient
tensors finite; learned-head serial path, cache1/chunk64, requested features intact.
Durable budget retains prior441.281998629seconds plus this active attempt, reserved
6758.718001371seconds; pause downtime is excluded. Active PID/birth13691/2974154,
start_monotonic29834.652367498/start_unix1791095277.2836585. GPU observed9.9GiB/73%.

The SAME heartbeat/Goal are ACTIVE. Continue candidate to natural5000evaluation,
then alternate reference/candidate at scheduled boundaries until each7200s cap;
reference remains evaluated5000/764.725s. No new checks/recipe/budget change or
2080host-control writes. The latest operator ledger is authoritative for live
commands/births/budget/next checkpoint; historical pause records below are superseded.

## Human RTX5080 resume — October 3, 23:21 PDT

The human requested “resume gpu usage”. Only RTX5080 is resumed for the existing
A8 comparison. The SAME recovery heartbeat is ACTIVE and native Goal is active;
sole operator `/root/gpu_supervisor` is reconnecting via tmux MCP using the fresh
shared host registry and checking availability before launch. No other chat is
contacted. RTX2080Ti controls remain untouched.

Continue candidate from exact step2081/cursor2084, archived checkpoint
`ac8af0f058b7d1d...`, settled budget441.281998629of7200 seconds; active_attempt null
at shutdown. Reference is preserved/evaluated at5000/764.725seconds. Pause downtime
is not training time. Reuse admitted native/model/zero/positive-resume evidence;
no preparation repeats, source/recipe/data/precision change, or unscheduled eval.
Resume SAME immutable source3bd/config/optimizer/RNG/cursor/recipe, observe actual
positive updates, then candidate5000 scheduled eval and alternate natural5000
boundaries until both independent7200s arms finish. Evaluation cap1200 unchanged.
Fresh process/resource/checkpoint availability and actual resumed updates are
pending operator evidence; the resume flag alone is not a launch claim.

## Final human-pause shutdown checkpoint — October 3, 18:34 PDT

Owned RTX5080 A8 usage is STOPPED. The sole operator issued the human-pause
`--stop` under its supervisor; candidate training exited0 at01:34:37.876UTC
(October3,18:34PDT), status stopped, step2081/cursor2084. Exact checkpoint
`ac8af0f058b7d1d...`, manifest `931a0fcc...`, optimizer/RNG/cursor exact; checkpoint,
budget, recipe and status hardlink-archived with verified inode. Settled candidate
budget441.281998629of7200seconds, active_attempt null,6758.718seconds remains.
Reference stays at evaluated5000/764.725seconds (6435.275remains). All progress
and raw evaluation failures preserved; comparison INCOMPLETE, Goal/monitor PAUSED.

Operator verified every recorded owned train/stop/monitor/native PID/group absent,
empty native process scan, empty compute-app query and no remote tmux session.
Host residual GPU usage was3537MiB/21% with no compute process; do not label that
residual host usage as an A8 job or claim the entire physicalGPU unused. Sole
operator confirmed SSH exited and only its local MCP session249 was killed;
transport is closed. Operator ledger contains final shutdown evidence.
No more remote activity after shutdown proof, no new GPU/training/eval/recovery
until a direct human resume. RTX2080Ti controls are untouched.

## Human RTX5080 pause — October 3, 18:34 PDT

Human requested “pause5080gpuusage”. Root set local rtx5080.pause_requested=true,
PAUSED the existing recovery heartbeat and native Goal, and directed the sole
operator to stop candidate training immediately, save/checkpoint exact state,
verify all owned groups/native contexts released, then disconnect. No further
training/evaluation/recovery or remote work beyond shutdown verification.
RTX2080Ti control is untouched. GPU release is pending actual operator proof;
the pause flag alone is not a resource-release claim.

Latest before shutdown: candidate1717/cursor1719,348.00of7200 trainer seconds,
33finite gradients, all requested features active. Reference preserved at evaluated
5000/764.725seconds; all checkpoints and raw failures remain retained. The comparison
is INCOMPLETE and paused, not marked achieved. Resume only on a new human request.

## Current execution checkpoint — October 3, 18:28 PDT

**Candidate REAL learned A8/all-nine midpoint training is running.** First proof
stopped gracefully at105/cursor105, checkpoint `ba2a6634...`, manifest `25121d47...`,
optimizer/RNG/cursor exact, hardlink-archived. Every enabled tensor had finite/
nonzero gradients and measured sampled movement: sign9/9, scale9/9, quantizer6/6,
midpoint9/9. Exact same-source/config resume advanced beyond105 to1200/cursor1201;
budget245.81s of7200, remaining6954.19s. New supervisor11472/birth1172206,
trainer11478/birth1172210 on boot517c4a36/source3bd; active reservation7171.1995s.
Actual33gradient tensors finite; cache1/chunk64, serial learned-head gradient path.
AdamW beta(.9,.999)/epsilon1e-8/clip1, latent0.1, signLR1e-3 and otherLR1e-5.

Reference is preserved at evaluated5000/cursor5009, budget764.725s (6435.275left),
exact checkpoint `884dd8c5...` archived. Scheduled eval PASSED/archived in705s;
report `cbd1bfe8...`, development loss5.1176, native acceptance0.0771 vsQ4_00.2659,
requestTPS81.48 vs135.49 (ratio0.6013), decoderatio0.5874. Five native timing
repetitions, fixed24development prompts, actual RTX5080/SM120/current9e2; no
sealed access. This is an interim below-Q4_0 result, not final acceptance/speed win.

Both zero evals and both native/model admissions passed. Reference exact resume
859→912 and candidate105→1200 are actual positive-step proofs; all budget/update
progress retained. Continue candidate to5000scheduled evaluation, then alternate
natural5000boundaries until both independent7200s caps/final native comparisons.
No extra105/859evals, no newgates/repeatedpreparation/science changes. A1 held out;
no2080control/query/experiment or unrelated fusion input. Sole operator-ledger
owns exact active artifacts/births/commands; root monitor remains ACTIVE.

## Standalone ownership claimed — October 3, 2026

Coordinator `01a103da-0980-7332-a041-3f95aca6a3f5` owns this goal exclusively.
The originating chat and interrupted workers are uninvolved. Existing heartbeat
`a8-qat-recovery-monitor` is retargeted here and ACTIVE at the same 15-minute
cadence. No duplicate monitor. RTX2080Ti, A1 and unrelated research stay paused.

New bounded team:

| Worker | Ownership | Preserved checkout |
|---|---|---|
| `/root/trainer_recovery`, Sol high | Trainer, prepared adapter, standalone evaluation, exact resume and cumulative budget | `/private/tmp/eagle-qat-a8-integration` |
| `/root/recipe_recovery`, Sol high | Matched recipes, effective configuration and parameter-family movement | `/private/tmp/eagle-qat-a8-recipe` |
| `/root/gpu_supervisor`, Luna high | Independent validation, sole tmux MCP/RTX5080 deployment and supervised runs | `/private/tmp/eagle-qat-a8-operator` |
| Root | Readiness review, integration, durable records and existing monitor | `/private/tmp/eagle-a8-standalone` |

Selected-lane readiness commit99a53ee is reviewed and adopted as a8c9986;
all50readiness CPU checks pass, including the6new selected-lane checks. Ownership
and readiness are integrated/published on main3e12ad6. Partial trainer and recipe
implementations are preserved by workera4a66fa and3d5fafa respectively.

A fourth bounded Sol worker `/root/native_request_metrics` owns a NEW native
request timing helper and tests, with trainer owner stitching its API into the
standalone evaluator. Existing tensor captures measure acceptance only; true
full-request timing must be measured separately within the same1200s evaluation
cap, with warmup/five repetitions and alternating variant order.

Fresh sole-operator read-only check confirms same WSLboot517c4a36, RTX5080/SM120,
zero utilization, no remote_job/trainer/native/pytest process or dedicated host
tmux sessions, approximately20.27GB available host RAM and357.34GB disk. Native
binaries and retained preparation-ready/zero-checkpoint artifacts are located;
prior receipt/config identities match. Remote main dirty source and untracked
checkouts remain untouched. No new model or optimizer has run.

Reference latent magnitude0.5 matches original dense-sign initialization;
candidate0.1 retains the same initial deployed signs/scales with zero midpoints
and initial learned clips. Next: complete/test integrated recipe+trainer, then
one decisive actual-model/native/resume validation and the fixed-budget matched
comparison. Failed post-checkpoint work must remain charged.

### Integrated review checkpoint

Recipe worker committed3d5fafa/20c2cad; root adopted23a0f5d/f8c504a.
Root21CPU recipe/cache-head checks plus Ruff/diff pass on Torch2.14.0.
Readiness50checks also pass on the project Torch2.14.0 runtime after supplying
the existing gguf-py module path. Recipe tests establish identical deployed
initial signs/scales; zero-affine CPU logit drift max1.19e-7 is bounded by
1e-6relative/2e-7absolute tolerance and identical argmax. No deployment failure
is inferred from that floating difference.

Root found the historical development evaluator uses frozenb4 while the enabled
learned/affine native support belongs to9e2. The team is adding explicit
`evaluation_native` metadata for current supported executable/libraries and
separately preserving original teacher/capture binary ancestry. Model/verifier
precision and request settings remain frozen. New native evaluation enables
shared packing and unused-head pruning for both arms; Torch cache/head audit
remains distinct, including the learned-head serial-gradient exception.

Crash-budget implementation preserves unresolved trainer charges and excludes
normal standalone evaluation/startup. Conservative uncertain downtime charging
on the same boot, or unresolved remaining reservation consumption after reboot,
must be disclosed if recovery uses it. New optional-family movement evidence is
measured by optimizer hooks, bounded samples and per-tensor finite gradients;
first-update float32 nonmovement during warmup cannot alone prove a broken
parameter family. A bounded early-update family gate retains actual proof.

### Matched gradient arithmetic correction

Sole-operator independent review found `a1_computation=single_forward` also
selects the alternate learned A8 VJP. The prior assumption that this flag is
irrelevant to A8 was wrong. Recipe commit2a3c489, adopted29ee3b3, selects
`reference` for both arms, preserving the agreed gradient baseline and all
requested candidate features. The execution exception for learned-head serial
training remains explicit. Cheap execution observations run each round; dense
recipe audits/packed sign telemetry run at initialization and diagnostic cadence.
Movement admission accumulates finite/nonzero gradients and measured displacement
for every enabled family across at most the first100budgeted updates, preserving
per-tensor evidence and exact checkpoint state.

Native timing helper49c7511 is adopted43ed317; its11CPUchecks are pending root
integration. The mandatory same-target no-speculation timing reference is being
added within the SAME1200s evaluation deadline; Q4_0 remains primary. No extra
training/evaluation budget or new research arm is selected.

### Integration ready for current-model execution

Root integrated trainerf30dad6/3272936 as5e3276a/a16bff5, on top of selected-lane
readiness, corrected recipe, accumulated movement, and native timing/target-only
work. Root82integrated CPUchecks plus7latest launch-contract checks pass on the
project Torch2.14.0 runtime with scripts/tests/src and populated gguf-py imports.
Ruff and diff checks pass. Worker independently reports167affected checks; all
actual GPU evidence is still pending. Current branch progress is published;
main integration publication is the next action before immutable deployment.

Prepared-data fixes preserve exact original receipt/data bytes: source6f's known
stages-wrapper audit hash may be reused only with a pinned historical receipt,
matching all actual data/runtime/dependency bytes, validated historical full-pass
provenance, and unchanged semantic auditor AST. The data child is initialized
under the original fixed teacher contract; actual current candidate linears,
optimizers and admission use the untouched candidate config. TRAIN and
standalone development use this explicit teacher/actor distinction.

Current gate provider: `train_prepared_continuous_w1ax:create_current_provider`.
Its CPU JSON record schema is `current_prepared_provider_v1`, with pinned arm
config locator, optional stages_manifest locator, original prepared_run_dir and
ready SHA, unique validation_run_dir and selected_shard0. Keep the production
spec.provider declaration frozen; pass this factory/binding via gate CLI flags.
Do not use the legacy `train_prepared --validate/--start` source6f launcher or its
`prepared_provider_binding_v1` emission for the new recipes.

Run sequence per arm: current native/model/backward/memory admissions; fresh
`train_continuous_w1ax --start --prepare-only --allow-cuda` with original prepared
corpus flags; native standalone step-zero development; exact resume into actual
7200-second training. First100budgeted updates prove enabled family movement;
then graceful `--stop` saves current state, verify group/context return and exact
`--resume` under UNCHANGED production config before continuing. No temporary
max_steps change or separate disposable trial. All updates remain charged to
the same arm. Scheduled5000/final evaluations retain the same1200s bounds.

Current evaluation runtime metadata is in ignored
`runs/qat-a8-recovery/evaluation-native-runtime-9e2.json`: supported9e2 executable
SHA1ca0c1d9..., exact immutable manifest/library pins and native_commit. This is
static identity evidence, not native CUDA readiness. Sole operator will record
actual process/run/source identities and effective CUDA deployment next.

### Sole-operator pipeline GO

Frozen execution source is published583480c79f3ca090de0952deca8278dcb7f15c2d;
main integration is8583b68. A concurrent independent CPU feature added files to
main; they were preserved, and the A8 deployment stays on the reviewed exact
source583480c. No old-chat coordination occurred. Final malformed-receipt/import
fix passed7root checks. The root integrated82checks and worker167affected checks
remain positive; no actual optimizer/GPU launch is inferred from CPU evidence.

Root authorized `/root/gpu_supervisor` to deploy that source into a new immutable
checkout, perform fresh source/runtime/resource/ownership checks, and progress
through native+model gates, each fresh step-zero native development evaluation,
first100budgeted realupdates/graceful stop/exact resume, and the matched7200s arms
without repeated root confirmation. Existing gates must pass; max2recoveries per
incident, raw failures preserved, exact owned groups/contexts released before
retry, no scientific/feature/budget fallback. The existing monitor remains ACTIVE.
The operator restored the unexpectedly clear RTX2080Ti local pause flag without
connecting to that host; this does not establish remote RTX2080Ti resource state.

### Deployment and worker retirement checkpoint

Trainer final TEST-ONLY1fdd126 is merged through5dd0a3d;10root A8integration
checks pass. Recipe/timing/readiness original histories are fully merged into
main89ad18c, with all runtime/config source bytes unchanged from frozen583480c.
After confirming clean tracked state and only disposable Python/Ruff caches,
root removed merged trainer, recipe, timing and readiness worktrees/local branches.
Their commits remain reachable on published main; no unfinished code was dropped.
The operator worktree and root integration worktree stay live for this run.

Sole operator fetched frozen583480c and created clean detached remote checkout
`/home/philip/binary-eagle-decoding/checkouts/a8-qat-run-583480c7`, native gitlink9e2;
old dirty remote main remains untouched. Fresh resource check still shows same
boot517c4a36, no project compute process and zero GPU utilization.

A CPU-only supervised disconnect probe is owned by that operator:
run`a8-cpu-disconnect-proof-20261003-01`, runtime session
`a8-cpu-idle-proof-20261003-01`, supervisor1676 and child group1682, started22:59:22UTC.
Reconnection after at least165s must verify identities then stop/verify that exact
probe group before GPU work. The initially proposed15–20minute wait was narrowed
to the established bounded proof; prior same-boot163.512s proof remains retained.
No new GPU model/optimizer is claimed. Local ignored registration carries these
identities and active sole ownership; heartbeat is unchanged and ACTIVE.

### Fresh CPU durability proof returned

Operator reconnected at23:03:53UTC after more than165seconds disconnected:
same boot517c4a36, supervisor1676 and child/group1682 remained live with the same
state. Exact tmux-session SIGINT then produced interrupted/exit-15 at23:04:00UTC;
both identities and runtime socket were absent, no GPU compute process, zero GPU
utilization. Raw proof remains under the immutable checkout's
`runs/a8-cpu-disconnect-proof-20261003-01`. This is durability/return evidence,
not model/GPU training success. The sole operator is materializing current arms
and native/model inputs under the existing GO.

Operational evidence retention: before later pruning, archive each completed raw
step-zero/scheduled/final evaluation and each step-zero/100-update proof checkpoint
under the arm's evidence-archive (same-filesystem hardlinks allowed), keeping
original bytes/hashes and an original-prefix-to-archive-prefix locator mapping.
This preserves matched initial controls and actual resume evidence without a new
source/config/budget change. Failed raw evaluation attempts remain protected by
source; large tensor JSONL is never treated as compact metadata collection.

### First native bootstrap and bounded input recovery

Reference zero-state CPU bootstrap completed23:14:25UTC/exit0 under dedicated
session`a8-ref-native-bootstrap-20261003-01`, supervisor2082/child2087. It loaded
original model snapshots and emitted only A8 (joint.npz873,206,016bytes), optimizer
updates0 and data-eligibility false. Source583480c, Torch2.14.0+cu130 and actual
RTX5080/SM120 identity are bound; this is CPU bootstrap, not optimizer training.

First native run command refused the selected `train-00000.jsonl`: that shard
contains prose without the gate's required code/reasoning domains. It failed
before native GPU execution. Original failure.json/command-0.log are retained;
operator verified no process/group, empty runtime tmux and unchanged idle GPU.
Recovery1/2 for incident`native_gate_missing_domains`: use the existing declared
balanced TRAIN gate_prompts with its frozen hash/membership, retain all original
failures and reuse the successful authenticated bootstrap bundle in a new
context/command attempt. Only gate prompt locator/output locations change; actual
training corpus/order, recipe, precision and budget remain unchanged. No repeated
model bootstrap or semantic corpus audit is authorized or needed.

For durable file partitioning root owns monitor-registration.json; sole operator
owns separate ignored operator-ledger.json/handoff evidence for actual commands,
process births/groups/sessions, checkpoints and incident attempts. Milestones
reconcile into the root registration and this goal file.

## Standalone chat handoff — October 3, 2026

The human corrected the execution structure: “just launch it as a seperate codex
task and dont touch it i dont want any cross incolvement”. The historical
coordinator chat has interrupted all four native workers and PAUSED its
`a8-qat-recovery-monitor`. It will create one standalone project chat with this
handoff, then perform no further coordination, messaging, monitoring or edits
for this goal. The new chat is authorized to own the goal/team, fix integration,
run the comparison and retarget/rewrite the paused heartbeat to itself. It must
claim ownership in STATUS/this file/ignored registration and use its own bounded
workers. Do not contact the old coordinator or reuse its interrupted workers.

Preserved implementation state (do not discard):

- `/private/tmp/eagle-qat-a8-integration`, branch
  `feature/qat-a8-integration-20261003`: uncommitted changes to continuous_qat.py,
  train_continuous_w1ax.py, train_prepared_continuous_w1ax.py and
  w1ax_continuous_stages.py, plus new test_a8_continuous_integration.py. Implements
  activation_bits=(8,), current authenticated PreparedProvider reuse and
  standalone evaluation request/result binding. Targeted tests were in progress.
- `/private/tmp/eagle-qat-a8-recipe`, branch `feature/qat-a8-recipe-20261003`:
  uncommitted configs/qat_a8_comparison.json and qat_recipe_audit.py. Recipe
  materialization, effective attachments/optimizer audit and telemetry draft;
  tests not yet verified. Preserve partial work and review before adoption.
- `/Users/pippo/github/binary-eagle-decoding-a8-readiness`, branch a8-readiness:
  clean committed `99a53ee3dfc6447566e23262c2a40c732ed8b9c7`, four owned files;
  selected-lane readiness/native collector, 59 focused CPU tests plus Ruff/diff
  pass. Not integrated into main. Independent integrated verification remains.
- `/private/tmp/eagle-qat-a8-operator`, branch
  `feature/qat-a8-operator-20261003`: clean at83a3153. Operator's fresh read-only
  evidence found retained corpus and original zero checkpoint intact. Registry
  address was reachable via tmux MCP; RTX5080 idle/no project compute process,
  WSL boot517c4a36-e475-4a5f-9fa6-65de57edc6fe, ~20.08GB host available and
  357.35GB disk. `.wslconfig` has20GB and instanceIdleTimeout=-1. Remote main has
  preexisting dirty source/untracked dirs: preserve and deploy new checkout.

The last operator observations did not start a CUDA model or optimizer. Fresh
availability must still be checked by the new sole operator; do not infer a
perpetual free-GPU claim. RTX5080 local pause flag was resumed for this authorized
goal, RTX2080Ti remains paused. The new team should verify no leftover owned
test/operator commands before acquiring the GPU; old worker interruption is not
a remote process teardown claim. Original data/checkpoints/raw failures and
untracked overnight research remain untouched.

Outstanding root review: ensure failed work since the last checkpoint remains
charged to a durable cumulative time budget; exact resume must not refund it.
Keep the documented learned-head serial-training exception. Do not repeat full
corpus semantic audit on each restart. Reuse existing operator machinery rather
than building a new chain of permission wrappers.

## Human authorization — October 3, 2026

“launch a team to do that for qat, fix it, then run with a monitor that will heal
if things break. starting with a8 first, we will hold out a1.” Implementation,
targeted tests, RTX5080 validation/training and bounded recovery are authorized
without another confirmation. This is the only active goal; the previous goal
remains historical and complete for its fixed reference recipe only.

Coordinator: chat `01a103da-0980-7332-a041-3f95aca6a3f5`. Historical ownership below is superseded by the standalone claim. No A1 model/optimizer/
training/evaluation; RTX2080Ti and unrelated architecture research remain paused.
Demonstrate the effective A8 recipe through actual updates, save/resume/export/
reload and native development comparison against Q4_0, including complete-request
throughput. Frozen target/verifier precision and sealed finals are unchanged.

## Selected experiment contract

- Reuse authenticated 10,000 TRAIN prompts / 3,899,930 supervised rows and
  unsealed development data. No recapture or repeated semantic auditing of trusted
  completed data; fresh integrity checks are allowed. Preserve token/feature/
  teacher/cache/mask/vocabulary ancestry.
- Two fresh A8-only arms with the same seed, data order, hard-CE objective,
  initial deployed signs/scales and update cadence. Preserve the original paired
  step1000 unchanged; do not reinterpret it as a new recipe's exact resume.
- Shared execution: enable cache/head optimization and actual supported A8
  computation savings. A1 single-forward controls are irrelevant to A8. Learned
  activation training currently requires serial head execution to preserve
  invocation-local LSQ gradients; disclose this deliberate effective-path
  difference and validate it rather than silently bypassing the quantizer.
- Control: fixed A8 activations, symmetric binary weights, baseline AdamW latent
  initialization and movement rules. Candidate: learned A8 activations, all-nine
  affine weight midpoints, existing lower-inertia latent magnitude 0.1, AdamW
  sign LR 0.001 / scale LR 0.00001 and norm clip 1. No direct-bit Bop, fusion
  correction, depth weighting, curriculum or trajectory refresh in this initial
  comparison. This tests a combined candidate, not isolated feature attribution.
- Initial training cap: 7,200 cumulative trainer-accounted seconds per arm,
  14,400 total, preserved across standalone evaluations and failure recovery.
  Each evaluation is separately bounded at 1,200 seconds. Evaluate both step-zero
  arms, scheduled checkpoints (proposed every 5,000 updates; save every 1,000)
  and final checkpoints; record actual overhead and distinct prompt/row coverage.
- Fail launch on requested/effective recipe, lane, module or optimizer mismatch.
  Measure sign/scale/midpoint/quantizer gradients and movement, sign flips/
  flip-backs/near-zero distances and effective cache/head execution. Current
  native/backward/memory/save/export/reload/positive-step resume proof is required
  for enabled options, with a short decisive integrated validation path.
- Standalone evaluation releases live trainer/model/optimizer before allocating
  evaluation resources. Preserve source/runtime/hardware/data/config/checkpoint
  identities and raw failures. Never treat evaluation elapsed time as serving
  throughput or CPU evidence as GPU performance.

## Team and ownership

Workers use isolated temporary worktrees and preserve other edits. Root reviews,
integrates and pushes tested commits and owns goal/status/decisions/monitor. No
concurrent GPU use or parallel edits to the same files.

| Worker | Owned responsibility | Checkout |
|---|---|---|
| `/root/a8_integration`, Sol high | Single-lane trainer/CLI, prepared reuse, standalone evaluator/resume and new tests | `/private/tmp/eagle-qat-a8-integration` |
| `/root/a8_recipe`, Sol high | New comparison configs, effective recipe audit, movement telemetry and tests | `/private/tmp/eagle-qat-a8-recipe` |
| `/root/a8_readiness`, Sol high | Existing readiness/native collector selected-lane adaptation and new tests | `/Users/pippo/github/binary-eagle-decoding-a8-readiness` |
| `/root/a8_operator`, Luna high | Independent verification and sole RTX5080 deployment/run/evaluation/recovery operator | `/private/tmp/eagle-qat-a8-operator` |

Consult docs/AGENT_OPERATIONS.md and relevant USER_LESSONS. Reuse integrated
evaluator reconstruction and host-save accounting; relevant research is
experiments/parallel20261002/sign_inertia/report.md. Preserve unmerged research
and ignored artifacts. SSH only through tmux MCP using the shared local registry.

## Monitor and healing contract

ACTIVE heartbeat `a8-qat-recovery-monitor`, every 15 minutes, is attached to this
coordinator and continues from this file and
ignored A8 monitor registration. Stay quiet on unchanged healthy/non-actionable
state; notify verified training start, meaningful milestones, failures/recovery,
completion or required user action. Idle agent turns and disconnected transport
do not establish job failure. New human pause/stop overrides recovery immediately.

On genuine failure, retain raw error and committed checkpoint, verify exact owned
process identities/groups and release GPU contexts before retry. At most two
automatic retries for one incident, charged to the original cumulative budget.
Resume optimizer/RNG/cursor/data/recipe exactly. Infrastructure reconnection and
resource-safe evaluation can recover without math changes. A concrete code bug
can be fixed with targeted tests and newly published source identity; record
lineage and prove resume compatibility. Never silently disable requested
features, increase budget, change precision/objective/seed/data, recapture or
pivot architectures. Persistent deterministic failure or an incompatible
scientific change requires a factual report and user decision.

Use one operator; never duplicate SSH/GPU operators or infer rotated addresses.
On completion verify resource release, publish evidence, pause heartbeat and
mark durable goal complete. Do not automatically archive the human chat.

## Initial launch checkpoint

Main baseline `83a3153`; four workers dispatched. Existing untracked overnight
research is preserved. Operator resumed only RTX5080 using the shared registry.
Fresh WSL boot and RTX5080/SM120 availability verified: no project GPU compute
process, 0% utilization, about 20.08 GB available host RAM and 357.35 GB disk.
Preexisting remote main changes are preserved; deploy a new immutable checkout.
Retained prepared corpus was located. No new CUDA model or optimizer has run.
The original below-Q4_0 result is historical and lacks a matched step-zero control.

Monitor creation succeeded. Registration is saved outside Git at
`runs/qat-a8-recovery/monitor-registration.json`. This local registration records
the worker identities, sole GPU owner, initial budget, retry cap and phase;
actual-model/optimizer flags remain false until observed evidence exists.

## Native request timing helper — standalone bounded feature

`/root/native_request_metrics` owns only the new
`scripts/a8_native_request_metrics.py` and focused tests in
`/private/tmp/eagle-a8-native-metrics` (branch
`feature/a8-native-metrics-20261003`). Existing trainer/stages files are untouched.
The deliverable reuses `benchmark_native_eagle` parsing/count aggregation and
`run_binary_head_capture.server_command` for the exact24development request
policy (F16 target/draft KV, greedy seed42, max128 outputs, draft5/pmin0).
Tensor capture and trace environment hooks are absent. Both native arms require
current `9e2c7a90051e738751aab7d7bd7c2d8201fb76e3` with shared pack and unused-head
pruning; historical teacher b4 source remains separate and is refused for timing.

Callable: `measure_a8_requests(sources, prompts, draft, output,
deadline=caller_absolute_monotonic_deadline, stop_file=STOP)`.
Five alternating repetitions each run both A8/Q4_0 with one warmup per server:
240measured requests and10warmups. The helper shares the evaluator's1200s
aggregate deadline and cannot start a new budget. Raw request/response/counts,
HTTP wall, server prefill/decode spans, IDs, kernel markers and mapped runtime
are retained. Aggregate count/time ratios and distributions lead with Q4_0;
TTFT is explicitly unavailable from the nonstreaming endpoint. Incomplete runs
publish no performance ratios, retain raw failures and exact teardown evidence.

Acceptance check:11focused CPU tests passed, including full synthetic contract,
count/time aggregation, frozen policy, historical-runtime rejection, no capture
hooks, deadline/STOP handling and real local owned-process-group cleanup without
killing an unrelated process. Ruff and diff checks pass. No SSH/GPU or measured
performance result. Remaining integration: trainer calls the helper after export
using derived current evaluation sources (explicit commit/env/runtime inventory),
rejects incomplete timing, binds its manifest and attaches actual hardware evidence;
sole Luna executes within the existing evaluation deadline.

## Independent study boundary note — October 4, 02:15 UTC

The independent DSpark/DFlash coordinator re-read STATUS/this goal after its
first compaction. A8 ownership, pause, optimizer/data/checkpoint budgets and
RTX5080 controls/jobs remain unchanged; no cross-chat contact occurred.

Human-authorized RTX2080Ti supported measurements are now complete: two phases,
1728 measured requests, six repeats, 2552.28/7200s inference. Released DFlash7/
DSpark7 request throughput improves 45.8%/42.6% over primary Q4 EAGLE and matches
all 144 primary outputs per arm. FFN-Q4 similarly wins its paired comparison;
phase-anchor variation is disclosed. All bounded diagnostics pass within their
scope, GPU and transport are closed, merged local worktrees removed. The
[separate report](../../experiments/dspark-sm75-20261003/results.md) preserves
methods/results and pending W1 implementation decision. This creates no new
active project goal or A8 action.
