import React, { useEffect } from 'react'
import './App.css'
import { Widget, addResponseMessage, addLinkSnippet } from 'react-chat-widget'

import 'react-chat-widget/lib/styles.css'
import logo from './ngb-living-logo.png'

function App() {
	const link = {
		title: 'Project Repository',
		link: 'https://github.com/LateNight01s/ngbliving_chatbot',
		target: '_blank',
	}

	async function postData(url = '', payload = {}) {
		const response = await fetch(url, {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json',
			},
			body: JSON.stringify(payload),
		})
		return response
	}

	async function handleNewUserMessage(msg) {
    const newMsg = { message: msg }
    await postData('http://localhost:8080', newMsg)
			.then((response) => response.text())
			.then((response) => {
				addResponseMessage(response)
				console.log(response)
			})
			.catch((error) => {
				console.log(error)
			})

		if (msg.includes('Project') || msg.includes('project')) {
			addLinkSnippet(link)
		} else if (msg.includes('image')) {
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
				profileAvatar={logo}
				handleNewUserMessage={handleNewUserMessage}
			/>
		</div>
	)
}

export default App
