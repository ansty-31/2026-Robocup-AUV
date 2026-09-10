[#ftl]
[#-- macro generate Processing List Header Start--]
[#macro generatePlHeader HspPlModel]
/**
 * @brief Record ${HspPlModel.name} Processing list (Processing List name : ${HspPlModel.name})
 * @param hsp_engine_context_t
 * @retval uint32_t
 */
uint32_t MX_HSP_SEQ_Record_PL_${HspPlModel.name}(hsp_engine_context_t *hmw)
{
#tuint32_t error = 0UL;
#thsp_core_status_t mw_status;
[@generateStreamBuffer HspPlModel /]
#t/* must use the PL id */  
#tmw_status = HSP_SEQ_StartRecordPL(hmw, ${HspPlModel.eventId});
#tif (mw_status != HSP_CORE_OK) error++;

[/#macro]
[#-- macro generate Processing List Header End--]

[#-- macro generate Processing List Close Start--]
[#macro generatePlClose HspPlModel]

#tmw_status = HSP_SEQ_StopRecordPL(hmw);
#tif (mw_status != HSP_CORE_OK) error++; 
[/#macro]
[#-- macro generate Processing List close End--]

[#macro generateStreamBuffer model]
    [#assign streamBufferAddr ="" ]
    [#if model.eventStreamBuffer]
        [#assign streamBufferName="stream_buffer_"+model.direction?lower_case  ]
        [#list model.libMethod as method ]
             [#assign name = method.name]
             [#assign args =""]
          [#if method.comment== "USE_STREAM_GET_BUFFER_ADDRESS"]
             [#if method.arguments??]
                 [#list method.arguments as argument]
                     [#if argument.genericType == "struct"]
                         [@generateStructCode argument/]
                         [#assign value =argument.name ]
                     [#else]
                         [#if !argument.value?? ]
                              [#continue]
                         [/#if]
                         [#assign value =argument.value ]
                     [/#if]

                     [#if argument.addressOf]
                         [#assign args+="&"+value ]
                     [#else]
                         [#assign args+=value ]
                     [/#if]
                     [#if argument?index != (method.arguments?size - 1)]
                         [#assign args+="," ]
                     [/#if]
                 [/#list]
             [/#if]
            #n#tvolatile uint32_t *${streamBufferName}_tmp = ${method.name}(${args});
            #n#tfloat32_t *${streamBufferName} = (float32_t*)${streamBufferName}_tmp;
            #n
         [/#if]

        [/#list]
    [/#if]
[/#macro]