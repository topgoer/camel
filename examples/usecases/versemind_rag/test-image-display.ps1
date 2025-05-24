# Test script to verify image display functionality in CAMEL - No Duplicate Images

Write-Host "Testing CAMEL image display functionality (no duplicates)..." -ForegroundColor Green

Write-Host "1. Make sure the backend server is running (start-backend.bat)."
Write-Host "2. Make sure the frontend server is running (start-frontend.bat)."
Write-Host "3. Open your browser to http://localhost:3200."
Write-Host "4. Upload an image using the paperclip icon in the chat interface."
Write-Host "5. Type a message and send it with the image."
Write-Host "6. Verify that the image appears only in the user message bubble."
Write-Host "7. Verify that the AI response appears below and references the image content but doesn't duplicate the image."

Write-Host "`nExpected behavior:" -ForegroundColor Cyan
Write-Host "- Image should appear in user message only"
Write-Host "- AI response should NOT contain the image"
Write-Host "- AI response text should still reference the image content"

Write-Host "`nIf the image appears in both user and AI messages or doesn't show at all, check the browser console for errors." -ForegroundColor Yellow
