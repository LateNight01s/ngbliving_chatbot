import React, { useEffect, useState } from 'react'
import './App.css'
import { Widget, addResponseMessage, addLinkSnippet } from 'react-chat-widget'

import 'react-chat-widget/lib/styles.css'

function App() {
	const [count, setCount] = useState(1)
	const link = {
		title: 'Project Repository',
		link: 'https://github.com/LateNight01s/ngbliving_chatbot',
		target: '_blank',
	}

	function handleNewUserMessage(message) {
		setCount(count + 1)
		console.log(message)
		addResponseMessage(`Message count: ${count}`)

		if (message.includes('Project') || message.includes('project')) {
			addLinkSnippet(link)
		} else if (message.includes('image')) {
			addResponseMessage('![prototype](https://i.imgur.com/10C1C9o.png)')
		}
	}

	useEffect(() => {
		addResponseMessage('Welcome to the NGB Living ChatBot')
		return () => {}
	}, [])

	return (
		<div className='App'>
			<header className='App-header'>
				<p>Welcome to NGB Living ChatBot</p>
			</header>
			<Widget
				title='NGB Living ChatBot'
				subtitle='First prototype'
				handleNewUserMessage={handleNewUserMessage}
			/>
		</div>
	)
}

export default App
