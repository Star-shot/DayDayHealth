"""
事件绑定模块
处理 UI 组件的事件响应
"""
import gradio as gr


def setup_events(components, handlers):
    """
    设置所有事件绑定
    
    Args:
        components: UI 组件字典
        handlers: 事件处理函数字典
    """
    # ==================== 数据处理事件 ====================
    
    # 上传数据后自动预览并分析
    components['data_file'].change(
        fn=handlers['load_preview_data'],
        inputs=components['data_file'],
        outputs=[components['data_output'], components['data_info']]
    ).then(
        fn=handlers['analyze_data'],
        inputs=components['data_file'],
        outputs=[
            components['report_markdown'],
            components['llm_prompt_state'],
            components['auto_missing_plot'],
            components['auto_outlier_plot'],
            components['auto_dist_plot'],
            components['auto_corr_plot']
        ]
    )
    
    # 单独分析按钮
    components['missing_btn'].click(
        fn=handlers['get_missing_info'],
        inputs=components['data_file'],
        outputs=[components['missing_info'], components['missing_plot']]
    )
    
    components['outlier_btn'].click(
        fn=handlers['get_outlier_info'],
        inputs=components['data_file'],
        outputs=components['outlier_plot']
    )
    
    components['dist_btn'].click(
        fn=handlers['get_distribution_info'],
        inputs=components['data_file'],
        outputs=components['dist_plot']
    )
    
    components['corr_btn'].click(
        fn=handlers['get_correlation_info'],
        inputs=components['data_file'],
        outputs=[components['corr_plot'], components['high_corr_df']]
    )
    
    # 预处理
    components['preprocess_btn'].click(
        fn=handlers['process_data'],
        inputs=[
            components['data_file'], 
            components['fill_strategy'], 
            components['outlier_method']
        ],
        outputs=[
            components['data_output'], 
            components['preprocess_output'], 
            components['report_markdown'], 
            components['llm_prompt_state'],
            components['auto_missing_plot'],
            components['auto_outlier_plot'],
            components['auto_dist_plot'],
            components['auto_corr_plot']
        ]
    )
    
    # 下载按钮
    components['download_data_btn'].click(
        fn=handlers['download_processed_data'],
        outputs=components['processed_file']
    )
    
    components['download_report_btn'].click(
        fn=handlers['download_report'],
        outputs=components['report_file']
    )
    
    # 发送给 LLM
    components['send_to_llm_btn'].click(
        fn=handlers['prepare_for_llm'],
        inputs=[components['llm_prompt_state']],
        outputs=[components['msg'], components['img_input']]
    )
    
    # ==================== 模型训练事件 ====================
    
    # 模型参数切换
    components['model_choice'].change(
        fn=handlers['toggle_params'],
        inputs=components['model_choice'],
        outputs=[
            components['rf_params'], 
            components['svm_params'], 
            components['lr_params']
        ]
    )
    
    # 训练按钮
    components['train_btn'].click(
        fn=handlers['train_model'],
        inputs=[
            components['train_file'], 
            components['model_choice'],
            components['rf_n_estimators'], 
            components['rf_max_depth'], 
            components['rf_max_features'],
            components['svm_kernel'],
            components['svm_C'],
            components['svm_gamma'],
            components['lr_penalty'],
            components['lr_C'],
            components['lr_solver']
        ],
        outputs=components['train_output']
    )
    
    # ==================== 模型评估事件 ====================
    
    components['eval_btn'].click(
        fn=handlers['evaluate_model'],
        inputs=components['eval_file'],
        outputs=[
            components['dataframe_component'], 
            components['roc_curve_plot'], 
            components['pr_curve_plot'], 
            components['confusion_matrix_plot']
        ]
    )
    
    # ==================== 批量预测事件 ====================
    
    components['pred_btn'].click(
        fn=handlers['make_prediction'],
        inputs=components['pred_file'],
        outputs=components['pred_output']
    )
    
    # ==================== 聊天事件 ====================
    
    # 模型选择变化时更新信息
    components['model_id'].change(
        fn=handlers['update_provider_info'],
        inputs=components['model_id'],
        outputs=components['provider_info']
    )
    
    # 回车发送
    components['msg'].submit(
        fn=handlers['user_input_handler'], 
        inputs=[
            components['msg'], 
            components['img_input'], 
            components['chatbot'], 
            components['image_cache']
        ], 
        outputs=[
            components['msg'], 
            components['img_input'], 
            components['chatbot'], 
            components['image_cache']
        ]
    ).then(
        fn=handlers['chat'], 
        inputs=[
            components['chatbot'], 
            components['model_id'], 
            components['image_cache']
        ], 
        outputs=components['chatbot']
    )
    
    # 按钮发送
    components['send_btn'].click(
        fn=handlers['user_input_handler'], 
        inputs=[
            components['msg'], 
            components['img_input'], 
            components['chatbot'], 
            components['image_cache']
        ], 
        outputs=[
            components['msg'], 
            components['img_input'], 
            components['chatbot'], 
            components['image_cache']
        ]
    ).then(
        fn=handlers['chat'], 
        inputs=[
            components['chatbot'], 
            components['model_id'], 
            components['image_cache']
        ], 
        outputs=components['chatbot']
    )
    
    # 导出对话历史
    components['download_history_btn'].click(
        fn=handlers['download_chat_history'],
        inputs=components['chatbot'],
        outputs=components['chat_history_file']
    )

