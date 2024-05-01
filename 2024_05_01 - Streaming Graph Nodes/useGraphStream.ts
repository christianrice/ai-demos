import { useState } from 'react'
import { getStreamResponse } from '@src/utils'

// Set up the interface for the Graph Stream
type NodeHistory = {
	node: string
	message: string
}

interface GraphState {
	isRunning: boolean
	shouldDisplay: boolean
	uiMessage: string | null
}

interface DefaultGraphNodeData {
	[key: string]: string
}

interface BaseGraphStreamState<NodeData = DefaultGraphNodeData> {
	currentNode: string | null
	nodeData: NodeData
	nodeHistory: NodeHistory[]
	graphState: GraphState
	finalOutput: string | null
}

// Set up the interface for the graph nodes we care about monitoring.
// We want to specify the nodes to observe, and a few properties for each node:
// - The formattedName is the name of the node as it appears in the stream
// export const productVisionGraph: ObservedGraphNodes = {
// 	'Product Vision Graph': {
// 		formattedName: 'Product Vision Graph',
// 		actionText: 'Analyzing your message',
// 		isGraphNode: true,
// 	},
// 	'extraction_classifier': {
// 		formattedName: 'Extraction Classifier',
// 		actionText: 'Analyzing your message ',
// 	},
// 	'extract_objectives': {
// 		formattedName: 'Extract Objectives',
// 		actionText: 'Extracting business objectives',
// 	},
// 	vision_rewriter: {
// 		formattedName: 'Vision Writer',
// 		actionText: 'Rewriting the vision statement',
// 		isFinalOutput: true,
// 	},
// }

type ObservedGraphNode = {
	formattedName: string
	actionText: string
	isGraphNode?: boolean
	isFinalOutput?: boolean
}

type ObservedGraphNodes = Record<string, ObservedGraphNode>


// Set up the interface for events streaming back from LangGraph
interface StreamingEventDataChunk {
	kwargs: {
		content: string
	}
}

interface StreamingEvent {
	event: string
	run_id: string
	name: string
	data?: {
		chunk?: StreamingEventDataChunk
	}
}


export const useGraphStream = () => {
	const defaultGraphState: BaseGraphStreamState = {
		graphState: {
			isRunning: false,
			shouldDisplay: false,
			uiMessage: null,
		},
		currentNode: null,
		nodeHistory: [],
		nodeData: {},
		finalOutput: null,
	}

	const [graphStream, setGraphStream] =
		useState<BaseGraphStreamState>(defaultGraphState)

	const resetGraphStream = () => {
		setGraphStream(defaultGraphState)
	}

	const handleJsonResponse = (jsonString: string) => {
		try {
			return JSON.parse(
				`[${jsonString.replace(/}{"event"/g, '},{"event"')}]`,
			) as StreamingEvent[]
		} catch (error) {
			console.error('Failed to parse JSON:', error, jsonString)
			return [] as StreamingEvent[]
		}
	}

	const handleStreamResponse = (
		api: string,
		inputString: string,
		graphNodes: ObservedGraphNodes,
	) => {
		getStreamResponse(api, inputString, value => {
			// Get the streamedEvents from the response
			const streamedEvents = handleJsonResponse(value)

			// Group events by consecutive run_id so that we can batch our state updates without missing node changes
			const groupedEvents: StreamingEvent[][] = []
			let currentGroup: StreamingEvent[] = []

			streamedEvents.forEach(event => {
				if (
					currentGroup.length === 0 ||
					event.run_id === currentGroup[currentGroup.length - 1].run_id
				) {
					currentGroup.push(event)
				} else {
					groupedEvents.push(currentGroup)
					currentGroup = [event]
				}
			})

			// Ensure the last group is added
			if (currentGroup.length > 0) {
				groupedEvents.push(currentGroup)
			}

			// Process each group of events
			groupedEvents.forEach(group => {
				setGraphStream(prevState => {
					const newState = { ...prevState }

					group.forEach(streamedEvent => {
						console.log(streamedEvent)
						const nodeConfig = graphNodes[streamedEvent.name]
						switch (streamedEvent.event) {
							case 'on_chain_start':
								if (nodeConfig) {
									// Every time a new node (even the main graph node) begins that we care about, update the following parameters:
									// - currentNode should be updated to the new node that started streaming
									// - isRunning should always be true until on_chain_end sets it to false
									// - shouldDisplay should always be true until on_chat_model_stream sets it to false for the isFinalOutput message.
									// - uiMessage should be updated to the current node's action text
									newState.currentNode = streamedEvent.name
									newState.graphState = {
										...newState.graphState,
										isRunning: true,
										shouldDisplay: true,
										uiMessage: nodeConfig.actionText,
									}
								}
								break
							case 'on_chain_end':
								if (nodeConfig && nodeConfig.isGraphNode) {
									// Now that the graph is complete, update the following parameters:
									// - isRunning should be set to false
									// - shouldDisplay should be set to false
									// - uiMessage will be updated should be set to false
									newState.graphState = {
										...newState.graphState,
										isRunning: false,
										shouldDisplay: false,
										uiMessage: 'Finished',
									}
								} else if (nodeConfig) {
									// Every time a subnode completes, update the node history with the completed message
									newState.nodeHistory = [
										...newState.nodeHistory,
										{
											node: streamedEvent.name,
											message: newState.nodeData[streamedEvent.name] || '',
										},
									]
								}
								break
							case 'on_chat_model_stream':
								if (newState.currentNode) {
									// Append the streamed content to the current node's data
									const currentData =
										newState.nodeData[newState.currentNode] || ''
									newState.nodeData[newState.currentNode] =
										currentData +
										(streamedEvent.data?.chunk?.kwargs?.content || '')

									if (
										graphNodes[newState.currentNode] &&
										graphNodes[newState.currentNode].isFinalOutput
									) {
										// If this is the finalOutput node:
										// - Update the content for finalOutput
										// - Set shouldDisplay to false
										// - Leave isRunning as true
										newState.finalOutput =
											(newState.finalOutput || '') +
											(streamedEvent.data?.chunk?.kwargs?.content || '')
										newState.graphState.shouldDisplay = false
									}
								}
								break
						}
					})
					return newState
				})
			})

			if (
				streamedEvents.some(
					streamedEvent =>
						streamedEvent.event === 'on_chain_end' &&
						graphNodes[streamedEvent.name]?.isGraphNode,
				)
			) {
				resetGraphStream()
			}
		})
	}

	return { graphStream, handleStreamResponse }
}
