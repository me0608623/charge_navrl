"""
Isaac Sim startup script: create a persistent rclpy-based /velodyne_points publisher.
Bypasses the OmniGraph ROS2PublishPointCloud node (which fails to create publisher in Isaac Sim 5.1).

Run via MCP execute_script after the scene is loaded and simulation is playing.
Re-run after any Isaac Sim restart.
"""
import omni.kit.app
import omni.graph.core as og
import numpy as np

LIDAR_GRAPH_PATH = "/World/charger_rover4_5_0/charger_rover_urdf5/velodyne_01"
TOPIC_NAME = "/velodyne_points"
FRAME_ID = "base_link"

# Remove existing subscription if any
if hasattr(omni.kit.app, '_velodyne_pub_handle'):
    omni.kit.app.get_app().get_update_event_stream().remove_subscription(
        omni.kit.app._velodyne_pub_handle)
    del omni.kit.app._velodyne_pub_handle

try:
    import rclpy
    from sensor_msgs.msg import PointCloud2, PointField

    if not rclpy.ok():
        rclpy.init()

    if not hasattr(omni.kit.app, '_velodyne_node'):
        omni.kit.app._velodyne_node = rclpy.create_node('isaac_velodyne_bridge')

    ros_node = omni.kit.app._velodyne_node
    pub = ros_node.create_publisher(PointCloud2, TOPIC_NAME, 10)
    omni.kit.app._velodyne_pub = pub

    graph = og.get_graph_by_path(LIDAR_GRAPH_PATH)

    def publish_velodyne(event):
        try:
            data = None
            sim_time = None
            for n in graph.get_nodes():
                pth = n.get_prim_path()
                if "read_lidar" in pth:
                    data = n.get_attribute("outputs:data").get()
                elif "read_simulation_time" in pth:
                    sim_time = n.get_attribute("outputs:simulationTime").get()

            if data is None or len(data) == 0:
                return

            msg = PointCloud2()
            if sim_time is not None:
                sec = int(sim_time)
                msg.header.stamp.sec = sec
                msg.header.stamp.nanosec = int((sim_time - sec) * 1e9)
            msg.header.frame_id = FRAME_ID
            msg.height = 1
            msg.width = len(data)
            msg.fields = [
                PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
            ]
            msg.is_bigendian = False
            msg.point_step = 12
            msg.row_step = 12 * len(data)
            msg.is_dense = True
            msg.data = np.array(data, dtype=np.float32).flatten().tobytes()
            pub.publish(msg)
            rclpy.spin_once(ros_node, timeout_sec=0)
        except Exception:
            pass

    handle = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(
        publish_velodyne, name="velodyne_rclpy_publisher"
    )
    omni.kit.app._velodyne_pub_handle = handle
    result = f"velodyne publisher started on {TOPIC_NAME}"

except Exception as e:
    result = f"ERROR: {e}"
