"""
Fix velodyne_01 ROS2PublishPointCloud QoS to VOLATILE durability.
Run this via Isaac Sim MCP once the scene (charger_rover_urdf5.usd) is loaded.

Usage via MCP execute_script, or save to stage on first run to make it persistent.

VOLATILE durability (durability=2) ensures late-joining subscribers (like MOT)
do NOT receive stale point cloud messages from previous simulation sessions,
preventing the TF extrapolation-into-past error.
"""
import omni.usd
import omni.graph.core as og

VOLATILE_QOS = '{"history": 1, "depth": 5, "reliability": 1, "durability": 2}'
# reliability 1 = RELIABLE, durability 2 = VOLATILE
# NOTE: BEST_EFFORT (reliability=2) causes publisher init failure in Isaac Sim 5.1

fixed = []
errors = []

stage = omni.usd.get_context().get_stage()
if stage is None:
    result = "ERROR: No stage loaded"
else:
    # Search all prims for ROS2PublishPointCloud nodes
    for prim in stage.Traverse():
        node_type_attr = prim.GetAttribute("node:type")
        if not node_type_attr:
            continue
        node_type = node_type_attr.Get()
        if node_type and "ROS2PublishPointCloud" in str(node_type):
            qos_attr = prim.GetAttribute("inputs:qosProfile")
            if qos_attr:
                old_val = qos_attr.Get()
                if qos_attr.Set(VOLATILE_QOS):
                    fixed.append(f"  {prim.GetPath()} : {old_val} → VOLATILE")
                else:
                    errors.append(f"  {prim.GetPath()} : Set() failed")
            else:
                # Try to create the attribute
                try:
                    qos_attr = prim.CreateAttribute("inputs:qosProfile", omni.usd.get_usd_context())
                    errors.append(f"  {prim.GetPath()} : no qosProfile attr")
                except Exception as e:
                    errors.append(f"  {prim.GetPath()} : {e}")

    if fixed:
        # Save stage so it persists
        omni.usd.get_context().save_stage()
        result = "FIXED and SAVED:\n" + "\n".join(fixed)
    elif errors:
        result = "ERRORS:\n" + "\n".join(errors)
    else:
        result = "No ROS2PublishPointCloud nodes found in stage"

print(result)
