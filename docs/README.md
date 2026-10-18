# Fleet hours: which robot logs are worth training on

A robot fleet records video and actions all day. Almost all of that time is transit. The hours that teach a world model are the rare ones: contact, a recovery, a maneuver the fleet seldom does. Teams already pay to collect the logs. They will pay again for a checksummed slice that stays inside a storage budget and can be rebuilt from its config.

## Approach

Each robot shift is one Kuration source, so a whole shift lands in train or in test. The synthetic fleet has four shifts. Most 3.2-second windows are transit. A few are contact. One contact window also teleports the camera.

The pipeline windows every shift, measures `contact_rate`, and scores the logged camera pose with `slam_motion_diversity` and `slam_quality`. Windows whose `slam_translation_jump_fraction` is above zero are dropped. `slam_quality_score` is the wrong flag here: one illegal step barely moves it, because the penalty is a fraction of the steps in the window.

`rare_event_coverage` shows the contact lift on a fixed count. It cannot take an hour budget, so the delivery uses the same idea with a budget: keep windows at or above 50% contact, bin them by path length, and `sample("equal", hours=...)`. `output` writes NPZ shards, `config.json`, and SHA-256 checksums.

## Result

On this fleet the pool is 22% contact. The count recipe keeps 6 windows, all contact. The hour budget fills exactly (12.8s train, 3.2s test) at 100% contact, and the same duration taken from the start of each shift is 0% contact. The teleport is absent from the shards. One robot is held out for test.

These labels are perfect and synthetic. The 100% figure is the contact filter working. It shows the SDK can fill a budget with the right windows. It does not show that contact can be inferred from video.

## Gaps

Contact must already be logged. The Counter-Strike collision operator needs a game mesh, so it is not used. `slam_reconstruct` from pixels needs the slam extra and a depth model; this run uses the pose the robot recorded. Pixel shift applies to keyboard presses, not a continuous drive command. `sample("weighted")` has no hour-budget path.
