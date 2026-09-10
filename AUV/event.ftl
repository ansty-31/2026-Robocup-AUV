[#ftl]
[#macro generateEvent HspPlModel]

    [#assign args = ""]
    [#list HspPlModel.libMethod as method ]
        [#assign name = method.name]
        [#assign args =""]
        [#if method.comment != "USE_STREAM_GET_BUFFER_ADDRESS"]
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
            #tmw_status = ${method.name}(${args});
            #tif (mw_status != HSP_CORE_OK) error++;
            #n
        [/#if]
   [/#list]

[#--#t${generate event }--]
[/#macro]


[#macro generateStructCode arg]
    [#if !arg.name?? || !arg.typeName??]
         [#return]
    [/#if]
    [#assign name =arg.name]
    [#assign type= arg.typeName]

    #t${type} ${name};#n
    [#list arg.argument as subArgument]
        [#if !subArgument.name?? || !subArgument.value??]
             [#continue]
        [/#if]
        #t${name}.${subArgument.name} = ${subArgument.value};
    [/#list]
[/#macro]