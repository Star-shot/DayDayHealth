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
    
    # 样例数据选择
    components['example_selector'].change(
        fn=handlers['select_example_data'],
        inputs=components['example_selector'],
        outputs=components['data_file']
    )
    
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
            components['auto_corr_plot'],
            components['viz_data_report']
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
            components['auto_corr_plot'],
            components['viz_data_report']
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
    
    # 传递到模型训练（带编码策略）
    components['send_to_train_btn'].click(
        fn=handlers['send_to_training'],
        inputs=[components['encode_strategy']],
        outputs=[components['train_file'], components['train_output']]
    )
    
    # ==================== 模型训练事件 ====================
    
    # 上传训练文件时自动获取列名
    components['train_file'].change(
        fn=handlers['get_file_columns'],
        inputs=components['train_file'],
        outputs=[components['feature_cols'], components['label_col']]
    )
    
    # 切分方式变化时切换显示
    components['split_method'].change(
        fn=handlers['toggle_split_params'],
        inputs=components['split_method'],
        outputs=[components['test_size'], components['k_folds']]
    )
    
    # 模型参数切换
    components['model_choice'].change(
        fn=handlers['toggle_params'],
        inputs=components['model_choice'],
        outputs=[
            components['rf_params'], 
            components['svm_params'], 
            components['lr_params'],
            components['xgb_params'],
            components['lgbm_params'],
            components['knn_params'],
            components['nb_params']
        ]
    )
    
    # 训练按钮（训练完成后自动评估）
    components['train_btn'].click(
        fn=handlers['train_model'],
        inputs=[
            components['train_file'],
            components['feature_cols'],
            components['label_col'],
            components['split_method'],
            components['test_size'],
            components['k_folds'],
            components['random_seed'],
            components['model_choice'],
            components['rf_n_estimators'], 
            components['rf_max_depth'], 
            components['rf_max_features'],
            components['svm_kernel'],
            components['svm_C'],
            components['svm_gamma'],
            components['lr_penalty'],
            components['lr_C'],
            components['lr_solver'],
            components['xgb_n_estimators'],
            components['xgb_max_depth'],
            components['xgb_learning_rate'],
            components['lgbm_n_estimators'],
            components['lgbm_max_depth'],
            components['lgbm_learning_rate'],
            components['lgbm_num_leaves'],
            components['knn_n_neighbors'],
            components['knn_weights'],
            components['knn_algorithm'],
            components['nb_type']
        ],
        outputs=[
            components['train_output'], 
            components['train_report_file'],
            components['train_report_preview'],
            components['viz_train_report'],
            components['viz_train_file']
        ]
    ).then(
        fn=handlers['evaluate_model'],
        inputs=None,  # 使用训练时保存的测试集
        outputs=[
            components['eval_status'],
            components['eval_metrics'],
            components['roc_curve_plot'], 
            components['pr_curve_plot'], 
            components['confusion_matrix_plot'],
            components['eval_report_file'],
            components['eval_report_preview'],
            components['viz_eval_report'],
            components['viz_eval_file']
        ]
    )
    
    # ==================== 模型评估事件 ====================
    
    # 手动评估按钮
    components['eval_btn'].click(
        fn=handlers['evaluate_model'],
        inputs=components['eval_file'],
        outputs=[
            components['eval_status'],
            components['eval_metrics'],
            components['roc_curve_plot'], 
            components['pr_curve_plot'], 
            components['confusion_matrix_plot'],
            components['eval_report_file'],
            components['eval_report_preview'],
            components['viz_eval_report'],
            components['viz_eval_file']
        ]
    )
    
    # ==================== 批量预测事件 ====================
    
    components['pred_btn'].click(
        fn=handlers['make_prediction'],
        inputs=components['pred_file'],
        outputs=components['pred_output']
    )
    
    # ==================== 聊天事件 ====================
    
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

