# MetalGoose service

The service uses the app's capture, engine, overlay, settings, HUD and shader sources. It has no settings window. GooseHUB controls it over a JSON-lines pipe.

```sh
python3 Service/build.py --output "$HOME/Library/Caches/GooseHUB/Build/MetalGoose"
```

`metalgoose-backend --status --json` reports the device, shader path and existing permissions without starting capture. `--serve` accepts commands on stdin and returns JSON on stdout. The first event is `ready`; EOF and `quit` stop the session and release its overlay and pointer constraint.

```json
{"id":"1","command":"status"}
{"id":"2","command":"start","pid":12345,"windowID":678}
{"id":"3","command":"configure","configuration":{"frameGenMode":"MGFG-1","frameGenMultiplier":2}}
{"id":"4","command":"stop"}
{"id":"5","command":"quit"}
```

`start` requires Screen Recording and Accessibility access. Permission requests are explicit `request-permissions` commands. The protocol and configuration fields are defined in `BackendProtocol.swift`. The GUI remains the normal Xcode app target; the service is built separately and shares its engine code.
