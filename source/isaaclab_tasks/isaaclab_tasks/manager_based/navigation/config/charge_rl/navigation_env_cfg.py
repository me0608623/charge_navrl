# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
導航環境配置文件 (Navigation Environment Configuration)
此文件定義了 ANYmal-C 機器人在導航任務中的完整配置，包括：
- 動作空間：使用預訓練的低層級策略
- 觀測空間：基座線性速度、投影重力、姿勢命令
- 獎勵函數：位置追蹤、方向追蹤、終止懲罰
- 命令生成：2D 姿勢命令（位置和朝向）
- 終止條件：超時、基座接觸
"""

import math

# 導入 Isaac Lab 核心模組
from isaaclab.envs import ManagerBasedRLEnvCfg  # 基於管理器的強化學習環境配置基類
from isaaclab.managers import EventTermCfg as EventTerm  # 事件項配置（用於重置等事件）
from isaaclab.managers import ObservationGroupCfg as ObsGroup  # 觀測組配置
from isaaclab.managers import ObservationTermCfg as ObsTerm  # 觀測項配置
from isaaclab.managers import RewardTermCfg as RewTerm  # 獎勵項配置
from isaaclab.managers import SceneEntityCfg  # 場景實體配置（用於定義傳感器等）
from isaaclab.managers import TerminationTermCfg as DoneTerm  # 終止項配置
from isaaclab.utils import configclass  # 配置類裝飾器，用於創建配置類
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR  # Isaac Lab 資源目錄路徑

# 導入導航任務相關的 MDP（馬可夫決策過程）模組
import isaaclab_tasks.manager_based.navigation.mdp as mdp
# 導入低層級環境配置（用於獲取機器人的基本配置）
from isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_c.flat_env_cfg import AnymalCFlatEnvCfg

# 載入低層級環境配置（包含機器人的基本設置，如關節配置、場景設置等）
# 這個配置會被用於獲取低層級策略所需的動作和觀測空間
LOW_LEVEL_ENV_CFG = AnymalCFlatEnvCfg()


@configclass
class EventCfg:
    """
    Configuration for events. / 事件配置類
    定義環境重置時的事件處理，主要用於設置機器人的初始狀態
    """

    # 重置基座狀態的事件配置
    # 在每次環境重置時，會根據指定的範圍隨機設置機器人的位置和速度
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,  # 使用均勻分佈重置根狀態的函數
        mode="reset",  # 模式：在重置時執行
        params={
            # 姿勢範圍（位置和朝向）
            # x, y: 機器人在水平面上的初始位置範圍（單位：米）
            # yaw: 機器人的初始朝向角度範圍（單位：弧度，-π 到 π 表示 360 度）
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            # 速度範圍（線性速度和角速度）
            # 所有速度都設為 0，表示機器人在重置時處於靜止狀態
            "velocity_range": {
                "x": (-0.0, 0.0),  # x 方向線性速度（前後）
                "y": (-0.0, 0.0),  # y 方向線性速度（左右）
                "z": (-0.0, 0.0),  # z 方向線性速度（上下）
                "roll": (-0.0, 0.0),  # 翻滾角速度（繞 x 軸旋轉）
                "pitch": (-0.0, 0.0),  # 俯仰角速度（繞 y 軸旋轉）
                "yaw": (-0.0, 0.0),  # 偏航角速度（繞 z 軸旋轉）
            },
        },
    )


@configclass
class ActionsCfg:
    """
    Action terms for the MDP. / MDP 動作項配置
    定義高層級導航策略的動作空間，這裡使用預訓練的低層級策略作為動作執行器
    這是一個分層控制架構：高層級策略輸出導航命令，低層級策略執行具體的運動控制
    """

    # 預訓練策略動作配置
    # 這是一個分層控制架構的關鍵組件：
    # - 高層級策略（導航策略）輸出目標位置和朝向
    # - 低層級策略（預訓練策略）接收高層級命令，並輸出關節位置控制
    pre_trained_policy_action: mdp.PreTrainedPolicyActionCfg = mdp.PreTrainedPolicyActionCfg(
        asset_name="robot",  # 機器人資產名稱
        # 預訓練策略的路徑（盲導航策略，不依賴視覺）
        # 這個策略已經學會了基本的運動控制，如行走、平衡等
        policy_path=f"{ISAACLAB_NUCLEUS_DIR}/Policies/ANYmal-C/Blind/policy.pt",
        # 低層級策略的降採樣倍數
        # 例如：如果高層級策略每 10 步執行一次，低層級策略每 4 步執行一次
        # 則低層級策略在一個高層級動作週期內會執行 10/4 = 2.5 次（實際為 2 次）
        low_level_decimation=4,
        # 低層級策略的動作空間：關節位置控制
        # 從低層級環境配置中獲取，定義了如何控制機器人的關節
        low_level_actions=LOW_LEVEL_ENV_CFG.actions.joint_pos,
        # 低層級策略的觀測空間
        # 從低層級環境配置中獲取，定義了低層級策略需要哪些觀測信息
        # 通常包括：基座速度、重力方向、關節狀態等
        low_level_observations=LOW_LEVEL_ENV_CFG.observations.policy,
    )


@configclass
class ObservationsCfg:
    """
    Observation specifications for the MDP. / MDP 觀測規格配置
    定義高層級導航策略的觀測空間，策略需要這些信息來做出導航決策
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """
        Observations for policy group. / 策略組的觀測配置
        定義高層級導航策略所需的所有觀測項
        注意：觀測項的順序會被保留，這對於神經網絡輸入很重要
        """

        # 觀測項（順序會被保留）
        # 1. 基座線性速度：機器人在世界座標系中的線性速度（x, y, z）
        #    用於讓策略了解機器人當前的運動狀態
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        
        # 2. 投影重力：重力向量在機器人基座座標系中的投影
        #    用於讓策略了解機器人的姿態（是否傾斜、是否翻倒）
        #    這是一個 3D 向量，表示重力方向相對於機器人的方向
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        
        # 3. 姿勢命令：當前生成的目標姿勢命令（目標位置和朝向）
        #    用於讓策略了解需要達成的目標
        #    這是一個相對命令，表示相對於當前位置的目標偏移
        pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "pose_command"})

    # 觀測組：將多個觀測項組織成一個組
    # 這裡只有一個策略組，包含上述三個觀測項
    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """
    Reward terms for the MDP. / MDP 獎勵項配置
    定義強化學習的獎勵函數，用於引導策略學習導航行為
    獎勵函數的設計直接影響策略的學習效果和最終行為
    """

    # 終止懲罰：當環境因為異常情況終止時給予的負獎勵
    # 這是一個強烈的負信號，鼓勵策略避免導致終止的行為（如翻倒、碰撞等）
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-400.0)
    
    # 位置追蹤獎勵（粗粒度）：鼓勵機器人接近目標位置
    # 使用 tanh 函數將位置誤差轉換為平滑的獎勵信號
    # std=2.0 表示在 2 米範圍內有較大的獎勵梯度，超出後獎勵趨於飽和
    # 這個獎勵項主要用於長距離導航，讓策略知道大致方向
    position_tracking = RewTerm(
        func=mdp.position_command_error_tanh,  # 使用 tanh 函數計算位置誤差獎勵
        weight=0.5,  # 獎勵權重
        params={"std": 2.0, "command_name": "pose_command"},  # 標準差為 2.0 米
    )
    
    # 位置追蹤獎勵（細粒度）：更精確的位置追蹤獎勵
    # std=0.2 表示在 0.2 米範圍內有較大的獎勵梯度
    # 這個獎勵項主要用於精確到達目標，讓策略在接近目標時更加精確
    # 兩個位置追蹤獎勵的組合可以讓策略既關注長距離導航，也關注精確到達
    position_tracking_fine_grained = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=0.5,
        params={"std": 0.2, "command_name": "pose_command"},  # 標準差為 0.2 米（更精確）
    )
    
    # 方向追蹤獎勵：鼓勵機器人朝向目標方向
    # 使用絕對誤差計算，負權重表示誤差越小獎勵越大（誤差越大懲罰越大）
    # 這確保機器人不僅到達目標位置，還要朝向正確的方向
    orientation_tracking = RewTerm(
        func=mdp.heading_command_error_abs,  # 計算朝向誤差的絕對值
        weight=-0.2,  # 負權重：誤差越大，獎勵越小（懲罰越大）
        params={"command_name": "pose_command"},
    )


@configclass
class CommandsCfg:
    """
    Command terms for the MDP. / MDP 命令項配置
    定義環境生成的任務命令，策略需要根據這些命令來執行導航任務
    命令會在每個回合開始時生成，並在指定時間後重新採樣
    """

    # 2D 姿勢命令配置：生成目標位置（x, y）和朝向（heading）
    # 這是導航任務的核心：策略需要學會根據這個命令導航到目標位置
    pose_command = mdp.UniformPose2dCommandCfg(
        asset_name="robot",  # 目標機器人資產名稱
        # 是否使用簡單朝向模式
        # False 表示使用完整的 2D 姿勢命令（位置 + 朝向）
        # True 表示只使用位置命令，朝向由位置決定
        simple_heading=False,
        # 命令重新採樣時間範圍（單位：秒）
        # (8.0, 8.0) 表示每 8 秒重新生成一個新的目標命令
        # 這意味著每個回合的持續時間為 8 秒
        resampling_time_range=(8.0, 8.0),
        # 是否啟用調試可視化
        # True 表示在仿真中可視化目標位置和朝向，方便調試和觀察
        debug_vis=True,
        # 命令生成範圍：定義目標位置和朝向的隨機生成範圍
        ranges=mdp.UniformPose2dCommandCfg.Ranges(
            pos_x=(-3.0, 3.0),  # x 方向位置範圍：-3 米到 3 米
            pos_y=(-3.0, 3.0),  # y 方向位置範圍：-3 米到 3 米
            heading=(-math.pi, math.pi),  # 朝向角度範圍：-π 到 π（360 度）
        ),
    )


@configclass
class TerminationsCfg:
    """
    Termination terms for the MDP. / MDP 終止項配置
    定義環境的終止條件，當滿足這些條件時，當前回合會結束
    終止條件用於處理異常情況，防止策略學習到不安全的行為
    """

    # 超時終止：當回合達到最大時間長度時終止
    # 這是一個正常的終止條件，表示回合自然結束
    # time_out=True 表示這是一個超時終止，不會觸發終止懲罰（但可能觸發其他終止條件的懲罰）
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    
    # 基座接觸終止：當機器人基座與地面或其他物體發生接觸時終止
    # 這是一個安全終止條件，防止機器人翻倒或發生不安全的接觸
    # 正常情況下，只有機器人的腳應該接觸地面，基座接觸通常表示機器人翻倒了
    base_contact = DoneTerm(
        func=mdp.illegal_contact,  # 檢查非法接觸的函數
        params={
            # 傳感器配置：指定要檢查的接觸力傳感器和要監控的物體
            # "contact_forces" 是接觸力傳感器名稱
            # "base" 是機器人基座的名稱，表示要監控基座的接觸
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"),
            # 接觸力閾值：當接觸力超過 1.0 牛頓時，觸發終止
            # 這個閾值用於過濾微小的接觸（如傳感器噪聲），只關注真正的接觸
            "threshold": 1.0,
        },
    )


@configclass
class NavigationEnvCfg(ManagerBasedRLEnvCfg):
    """
    Configuration for the navigation environment. / 導航環境配置類
    這是導航環境的主要配置類，整合了所有子配置（動作、觀測、獎勵、命令、終止等）
    繼承自 ManagerBasedRLEnvCfg，提供了基於管理器的強化學習環境的基礎功能
    """

    # ==================== 環境設置 ====================
    # 場景配置：從低層級環境配置中獲取，包含機器人、地面、傳感器等場景元素
    scene: SceneEntityCfg = LOW_LEVEL_ENV_CFG.scene
    # 動作配置：定義高層級策略的動作空間（使用預訓練低層級策略）
    actions: ActionsCfg = ActionsCfg()
    # 觀測配置：定義高層級策略的觀測空間
    observations: ObservationsCfg = ObservationsCfg()
    # 事件配置：定義環境重置時的事件處理
    events: EventCfg = EventCfg()
    
    # ==================== MDP 設置 ====================
    # 命令配置：定義任務命令的生成方式（目標位置和朝向）
    commands: CommandsCfg = CommandsCfg()
    # 獎勵配置：定義強化學習的獎勵函數
    rewards: RewardsCfg = RewardsCfg()
    # 終止配置：定義環境的終止條件
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        """
        Post initialization. / 後初始化方法
        在配置類實例化後自動調用，用於設置一些依賴於其他配置的參數
        這裡主要設置仿真時間步長、降採樣倍數、回合長度等關鍵參數
        """

        # 設置仿真時間步長：從低層級環境配置中獲取
        # 這決定了物理仿真的精度，通常為 0.002 秒（500 Hz）
        self.sim.dt = LOW_LEVEL_ENV_CFG.sim.dt
        
        # 設置渲染間隔：從低層級環境配置的降採樣倍數獲取
        # 這決定了視覺渲染的頻率，通常不需要每個物理步都渲染
        self.sim.render_interval = LOW_LEVEL_ENV_CFG.decimation
        
        # 設置高層級策略的降採樣倍數：低層級降採樣的 10 倍
        # 例如：如果低層級降採樣為 4，則高層級降採樣為 40
        # 這意味著高層級策略每 40 個物理步執行一次動作
        # 這種設計允許高層級策略以較低的頻率運行，專注於導航決策
        self.decimation = LOW_LEVEL_ENV_CFG.decimation * 10
        
        # 設置回合長度（單位：秒）：從命令重新採樣時間獲取
        # 這決定了每個訓練回合的持續時間，與命令重新採樣時間一致
        self.episode_length_s = self.commands.pose_command.resampling_time_range[1]

        # 如果場景中有高度掃描器（用於地形感知），設置其更新週期
        # 更新週期設置為低層級策略的執行週期，確保高度信息與低層級策略同步
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = (
                self.actions.pre_trained_policy_action.low_level_decimation * self.sim.dt
            )
        
        # 如果場景中有接觸力傳感器，設置其更新週期為仿真時間步長
        # 接觸力需要高頻率更新，以便及時檢測碰撞和接觸事件
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt


class NavigationEnvCfg_PLAY(NavigationEnvCfg):
    """
    用於遊玩/測試模式的導航環境配置類
    繼承自 NavigationEnvCfg，但針對實際運行和測試進行了優化：
    - 減少環境數量以提高性能
    - 增大環境間距以避免視覺干擾
    - 禁用觀測噪聲以獲得穩定的性能表現
    """
    
    def __post_init__(self) -> None:
        """
        後初始化方法：先調用父類的初始化，然後進行遊玩模式的特定設置
        """
        # 調用父類的後初始化方法，設置基本的仿真參數
        super().__post_init__()

        # ==================== 遊玩模式優化設置 ====================
        # 減少環境數量：從默認值（可能是 4096 或更多）減少到 50
        # 這可以顯著提高渲染性能，使實時可視化成為可能
        # 在訓練時使用大量並行環境可以加速訓練，但在測試時不需要那麼多
        self.scene.num_envs = 50
        
        # 增大環境間距：從默認值增大到 2.5 米
        # 這確保不同環境中的機器人不會在視覺上重疊，方便觀察單個機器人的行為
        # 較大的間距也有助於避免環境之間的干擾
        self.scene.env_spacing = 2.5
        
        # 禁用觀測噪聲：關閉觀測空間的隨機化/噪聲
        # 在訓練時，觀測噪聲可以提高策略的魯棒性（域隨機化）
        # 在測試時，禁用噪聲可以獲得穩定、可重現的性能表現
        self.observations.policy.enable_corruption = False
