# For Running Simulator:

## Commands:
Open Terminal in the location where 'rtde_control_loop.urp' exists
```bash
docker run --rm -dit -p 30004:30004 -p 5900:5900 -p 6080:6080 universalrobots/ursim_e-series
docker cp rtde_control_loop.urp <container_id>:/ursim/programs/rdte_control_loop.urp
```