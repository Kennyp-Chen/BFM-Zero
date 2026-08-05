# PiPlus S-12L8A0G2H0W asset

This directory vendors the PiPlus S-12L8A0G2H0W hardware description used by
the `piplus_soccer` task in the `env_soccer` environment.

- Source package: `/localhdd/Projects/ht_urdf/ht_urdf/PiPlus_S_12L8A0G2H0W`
- Source URDF: `PiPlus_S_12L8A0G2H0W/urdf/PiPlus_S_12L8A0G2H0W.urdf`
- Source MJCF: `PiPlus_S_12L8A0G2H0W/xml/PiPlus_S_12L8A0G2H0W.xml`
- Training MJCF: `xml/piplus_h0w_bfm.xml`
- Source URDF SHA-256: `65ea308b35584caeba2bdf193a2886a56226410e327e31a538641cacffc9a7ad`
- Source MJCF SHA-256: `13bba061c181c58c47c51fc1fbd0b94d6ecc70d6825084e3630cb6b5b6264127`
- Vendored URDF SHA-256: `20b8ebc688b1244b917c707a0e63dab2fae22284cd466c18b135a973740efdd7`
- Training MJCF SHA-256: `362a5b1bc1b2d6962a448c65672e9b48567a2177c072ea57719f0d25b8db0ad3`

The training MJCF keeps the source body geometry, inertias, joint ranges, and
collision geometry. It changes only the floating root to `<freejoint>`, removes
Isaac/URDF-only `actuatorfrcrange` attributes, and adds 22 same-name MuJoCo
motors in the motion-data joint order required by MotionLib.

The vendored URDF differs from the source only by replacing ROS `package://`
mesh URIs with paths relative to `urdf/`, so it resolves entirely within this
asset directory.
